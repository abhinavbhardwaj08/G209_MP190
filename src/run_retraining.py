
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.feature_engineering import build_preprocessor
from src.config import load_params
from src.retrain import split_later_data, evaluate_promotion_rule
from src.evaluate import calculate_mae

features = [
    "pm25_today", "wind_kmph", "humidity_pct",
    "temp_c", "traffic_index", "stubble_fires",
    "day_type", "station_zone"
]

target = "pm25_next_day"

original = pd.read_csv("data/processed/train_cleaned.csv")
later = pd.read_csv("later.csv")

new_train, evaluation = split_later_data(later)

combined = pd.concat([original, new_train], ignore_index=True)

X_train = combined[features]
y_train = combined[target]

X_test = evaluation[features]
y_test = evaluation[target]

v1 = joblib.load("models/model.joblib")

numeric = features[:6]
categorical = features[6:]

preprocessor = build_preprocessor(numeric, categorical)
params = load_params()["model"]["hyperparameters"]

v2 = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(**params))
])

v2.fit(X_train, y_train)

mae_v1 = calculate_mae(y_test, v1.predict(X_test))
mae_v2 = calculate_mae(y_test, v2.predict(X_test))

result = evaluate_promotion_rule(mae_v1, mae_v2)

print("V1 MAE:", mae_v1)
print("V2 MAE:", mae_v2)
print("Relative improvement:", result["relative_improvement"])
print("Decision:", result["decision"])

joblib.dump(v2, "models/model_v2.joblib")
