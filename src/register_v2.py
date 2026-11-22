
import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow_fresh.db")
mlflow.set_experiment("mp190_kochi_air_quality")

model = joblib.load("models/model_v2.joblib")
client = MlflowClient()

with mlflow.start_run(run_name="retrained_v2_registered") as run:
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        serialization_format="cloudpickle"
    )

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name="mp190-pm25-predictor"
    )

    print("New V2 version:", registered.version)
    print("V2 run ID:", run.info.run_id)

    client.set_registered_model_alias(
        name="mp190-pm25-predictor",
        alias="champion",
        version=registered.version
    )

    print("Champion updated to version:", registered.version)
