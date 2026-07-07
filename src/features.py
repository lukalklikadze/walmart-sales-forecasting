"""Feature engineering as sklearn-compatible transformers.

Two independent groups (both operate on the RAW joined frame from load_raw):

  Group A — CalendarFeatures: calendar/holiday/markdown/static features.
    Stateless, row-independent. Used by ALL model families.

  Group B — LagFeatures: lags + rolling stats of Weekly_Sales per (Store, Dept).
    Stateful: fit() snapshots the training history so transform() can compute
    lags for raw test rows. Used by TREE models only (deep nets skip it).

  DropColumns: tiny helper to drop the raw Date column (and anything else
    non-numeric) right before the regressor.

Everything returns a DataFrame, so column names survive through a Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]
GROUP_COLS    = ["Store", "Dept"]

# Named LagFeatures configs — log BOTH as MLflow runs so the contrast is recorded:
#   BASELINE: the naive short-lag spec. Scores great in-sample but fails on a
#     long horizon (39-week holdout WMAE ~10.2k) because lag_1/lag_2/rollings
#     are unobservable over ~97% of the forward test block, yet the model
#     learns to lean on them.
#   SUBMISSION: horizon-safe long lags only. lag_52/53 reach back into training
#     history for the ENTIRE 39-week test block (39-week holdout WMAE ~2.1k).
# Usage: LagFeatures(**LAG_CONFIG_SUBMISSION)
LAG_CONFIG_BASELINE   = dict(lags=(1, 2, 52), windows=(4, 12))
LAG_CONFIG_SUBMISSION = dict(lags=(52, 53, 104), windows=())


class CalendarFeatures(BaseEstimator, TransformerMixin):
    """Group A: calendar, per-holiday, markdown and static-store features.

    Adds:
      - Year, Month, WeekOfYear (ISO), week_sin/week_cos (cyclical week-of-year)
      - is_superbowl / is_laborday / is_thanksgiving / is_christmas_flagged
        (IsHoliday split by month: Feb / Sep / Nov / Dec)
      - is_week_before_christmas: the Friday falling on Dec 18-24, i.e. the
        week where the real Christmas shopping happens. Kaggle does NOT flag
        this week as a holiday, but per EDA it carries the true Dec spike.
      - md{n}_recorded flags (1 = MarkDown{n} was present), then MarkDown NaN->0
        so the model can distinguish "not recorded" from a genuine zero.
      - Type one-hot (Type_A/B/C, fixed categories so columns are stable),
        raw Type column dropped. Size and the macro columns pass through.
      - IsHoliday cast to int.

    Stateless (fit is a no-op), so train/test transforms are identical.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # date parts + cyclical week-of-year (week 53 wraps onto week 1)
        df["Year"]       = df["Date"].dt.year
        df["Month"]      = df["Date"].dt.month
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        angle          = 2.0 * np.pi * (df["WeekOfYear"] - 1) / 52.0
        df["week_sin"] = np.sin(angle)
        df["week_cos"] = np.cos(angle)

        # per-holiday flags: the 4 flagged weeks are uniquely identified by month
        hol = df["IsHoliday"].astype(bool)
        df["is_superbowl"]          = (hol & (df["Month"] == 2)).astype(int)
        df["is_laborday"]           = (hol & (df["Month"] == 9)).astype(int)
        df["is_thanksgiving"]       = (hol & (df["Month"] == 11)).astype(int)
        df["is_christmas_flagged"]  = (hol & (df["Month"] == 12)).astype(int)
        # the Friday on Dec 18-24 is always the week preceding the flagged
        # Christmas week (flagged Fridays fall on Dec 25-31)
        df["is_week_before_christmas"] = (
            (df["Month"] == 12) & df["Date"].dt.day.between(18, 24)
        ).astype(int)
        df["IsHoliday"] = hol.astype(int)

        # markdowns: NaN means "not recorded" (absent before ~Nov 2011), not zero
        for col in MARKDOWN_COLS:
            df[f"md{col[-1]}_recorded"] = df[col].notna().astype(int)
            df[col] = df[col].fillna(0.0)

        # Type one-hot with fixed categories so train/test columns always match
        for t in ("A", "B", "C"):
            df[f"Type_{t}"] = (df["Type"] == t).astype(int)
        df = df.drop(columns=["Type"])

        return df


class LagFeatures(BaseEstimator, TransformerMixin):
    """Group B: lag and rolling features of Weekly_Sales per (Store, Dept).

    Adds lag_{k} for k in `lags` and roll{w}_mean / roll{w}_std for w in
    `windows`. All features only ever look strictly BEFORE the row's own date,
    so there is no leakage of the row's own or future target values.

    How it works
    ------------
    fit(X, y) snapshots the training history as (Store, Dept, Date, sales).
    transform(X) then computes, for every row in X:

      - lag_k: exact-date lookup — the history value at Date - k*7 days for the
        same (Store, Dept). Dates are all Fridays, so alignment is exact.
      - roll{w}: the history is unioned with X's (Store, Dept, Date) keys
        (unseen keys get NaN sales), sorted by Date within each series, the
        sales are shift(1)-ed, and rolling stats are taken over the previous
        w OBSERVED weeks. pandas skips NaNs inside the window.

    Test-time semantics: test rows pull lags from the stored TRAIN history.
    The test block sits right after train, so the first test week gets real
    lag_1/lag_2 values; deeper test weeks have no recorded sales 1-2 weeks
    back and get NaN. lag_52 reaches 52 weeks back into train and therefore
    exists for the ENTIRE 39-week test block — it is the workhorse feature at
    predict time. Rolling stats likewise fade to NaN as the window drifts past
    the train/test boundary. Use a NaN-tolerant model (LightGBM, XGBoost,
    HistGradientBoosting) after this step.

    fit() must see only training data — inside a Pipeline under cross-validation
    sklearn refits it per fold, which keeps the history leakage-free.

    If y is not passed to fit(), a Weekly_Sales column in X is used instead.
    """

    def __init__(self, lags=(1, 2, 52), windows=(4, 12)):
        self.lags = lags
        self.windows = windows

    def fit(self, X, y=None):
        if y is None:
            if "Weekly_Sales" not in X.columns:
                raise ValueError(
                    "LagFeatures.fit needs y (or a Weekly_Sales column in X)."
                )
            y = X["Weekly_Sales"]
        hist = X[GROUP_COLS + ["Date"]].copy()
        hist["_sales"] = np.asarray(y, dtype=float)
        self.history_ = (
            hist.drop_duplicates(GROUP_COLS + ["Date"])
                .sort_values(GROUP_COLS + ["Date"])
                .reset_index(drop=True)
        )
        return self

    def transform(self, X):
        if not hasattr(self, "history_"):
            raise RuntimeError("LagFeatures must be fitted before transform.")
        out  = X.copy()
        idx  = out.index
        keys = GROUP_COLS + ["Date"]

        # ---- exact-date lags: shift history forward k weeks, left-merge
        for k in self.lags:
            lagged = self.history_.rename(columns={"_sales": f"lag_{k}"})
            lagged = lagged.assign(Date=lagged["Date"] + pd.Timedelta(weeks=k))
            out = out.merge(lagged, on=keys, how="left")
        out.index = idx  # left-merge on unique keys preserves row order

        # ---- rolling stats over the previous `w` observed weeks
        combined = self.history_.merge(
            out[keys].drop_duplicates(), on=keys, how="outer"
        ).sort_values(keys)
        combined["_prev"] = combined.groupby(GROUP_COLS)["_sales"].shift(1)
        grp = combined.groupby(GROUP_COLS)["_prev"]
        roll_cols = []
        for w in self.windows:
            combined[f"roll{w}_mean"] = grp.transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            combined[f"roll{w}_std"] = grp.transform(
                lambda s: s.rolling(w, min_periods=2).std()
            )
            roll_cols += [f"roll{w}_mean", f"roll{w}_std"]

        out = out.merge(combined[keys + roll_cols], on=keys, how="left")
        out.index = idx
        return out


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop columns the regressor can't ingest (by default the raw Date).

    Columns absent from X are silently ignored, so the same instance works
    in both Group A-only and Group A+B pipelines.
    """

    def __init__(self, columns=("Date",)):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])
