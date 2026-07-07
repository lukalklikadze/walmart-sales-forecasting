import os
import mlflow

# ---- shared
GITHUB_URL   = "https://github.com/lukalklikadze/walmart-sales-forecasting.git"
REPO_OWNER   = "llikl23"   # DagsHub username — MLflow server lives here (same for both teammates)
REPO_NAME    = "walmart-sales-forecasting"
KAGGLE_COMP  = "walmart-recruiting-store-sales-forecasting"
DATA_DIR     = "/content/data"
TRACKING_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"

def setup_env():
    """Load THIS user's own secrets, point MLflow at the shared DagsHub server."""
    from google.colab import userdata
    os.environ["KAGGLE_USERNAME"]          = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"]               = userdata.get("KAGGLE_KEY")
    os.environ["MLFLOW_TRACKING_USERNAME"] = userdata.get("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = userdata.get("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(TRACKING_URI)
    return TRACKING_URI
