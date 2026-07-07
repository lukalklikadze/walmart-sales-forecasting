def add_features(df):
    """Starter only — expand together after EDA."""
    df = df.copy()
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)
    # TODO (joint): lag features (sales 1/2/52 weeks ago), rolling mean/std,
    #   holiday flags, MarkDown NA handling, encode Store/Dept/Type.
    # Design goal: expose as sklearn transformers so the saved model runs on the
    #   RAW test set (the assignment's Pipeline requirement).
    return df
