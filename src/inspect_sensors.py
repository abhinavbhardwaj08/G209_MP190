
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df = df.drop_duplicates()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["station_zone", "date"])

columns = [
    "pm25_today",
    "wind_kmph",
    "humidity_pct",
    "temp_c",
    "traffic_index",
    "stubble_fires"
]

for zone, group in df.groupby("station_zone"):
    print(f"\nZONE: {zone}")

    for column in columns:
        runs = group[column].ne(group[column].shift()).cumsum()
        longest = group.groupby(runs).size().max()
        print(f"{column}: {longest}")
