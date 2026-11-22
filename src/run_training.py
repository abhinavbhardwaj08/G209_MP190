
import pandas as pd
from sklearn.model_selection import train_test_split

from src.train import train_model

df = pd.read_csv("data/processed/train_cleaned.csv")

features = [
    "pm25_today",
    "wind_kmph",
    "humidity_pct",
    "temp_c",
    "traffic_index",
    "stubble_fires",
    "day_type",
    "station_zone",
]

X = df[features]
y = df["pm25_next_day"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

result = train_model(X_train, y_train, X_val, y_val)

print("Training status:", result["status"])
print("MLflow run ID:", result["run_id"])
print("Evaluation metrics:", result["metrics"])
