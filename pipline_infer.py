# pipeline_infer.py
import pandas as pd
import numpy as np
import boto3
import pickle
import time
from math import radians, sin, cos, sqrt, atan2

S3_BUCKET = "evalon-ai-team-1"
RAW_PREFIX = "raw/"
MODEL_PREFIX = "models/"
RESULT_PREFIX = "results/"
RESULT_FILE = RESULT_PREFIX + "rf_top50_each_city.csv"

s3 = boto3.client("s3")

CITY_CENTERS = {
    "Barcelona": (41.3874, 2.1686),
    "Madrid":    (40.4168, -3.7038),
    "Valencia":  (39.4702, -0.3768),
}

# --------------------------------------------
# REAL DISTANCE FORMULA → returns METERS
# --------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# --------------------------------------------
# Distance to each city center in meters
# --------------------------------------------
def calculate_distance(row):
    city = row.get("city")
    lat = row.get("latitude")
    lon = row.get("longitude")

    if city not in CITY_CENTERS or pd.isna(lat) or pd.isna(lon):
        return np.nan

    c_lat, c_lon = CITY_CENTERS[city]
    return haversine(lat, lon, c_lat, c_lon)  # METERS


# --------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------
def clean_and_engineer(df):
    print("[*] Cleaning & feature engineering...")
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    num_cols = ["price", "constructedarea", "bathnumber", "roomnumber",
                "latitude", "longitude", "builtyear"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # distance calculation to center (meters)
    print("    - Calculating distance to city center (meters)...")
    df["distance_to_center"] = df.apply(calculate_distance, axis=1)
    df["distance_to_center"].fillna(df["distance_to_center"].median(), inplace=True)

    # distance_to_metro already in meters → DO NOT TOUCH
    if "distance_to_metro" in df.columns:
        df["distance_to_metro"] = pd.to_numeric(df["distance_to_metro"], errors="coerce")

    # extra features
    print("    - Creating ratio features...")
    df["price_m2"] = df["price"] / df["constructedarea"]
    df["rooms_per_m2"] = df["roomnumber"] / df["constructedarea"]
    df["bathrooms_per_room"] = df["bathnumber"] / df["roomnumber"].replace(0, np.nan)

    print("    - Filling missing values...")
    df = df.fillna(df.median(numeric_only=True)).fillna("Unknown")

    print("[+] Feature engineering complete.")
    return df


def get_features(df):
    return [c for c in [
        "constructedarea","bathnumber","roomnumber","distance_to_center",
        "price_m2","rooms_per_m2","bathrooms_per_room","builtyear",
        "propertytype","neighborhood","district"
    ] if c in df.columns]


def load_model(city):
    local_path = f"/tmp/model_{city}.pkl"
    s3_key = MODEL_PREFIX + f"model_{city}.pkl"

    print(f"[*] Downloading model for {city}...")
    s3.download_file(S3_BUCKET, s3_key, local_path)
    print(f"    - Model downloaded: {local_path}")

    return pickle.load(open(local_path, "rb"))


def run_inference(upload_filename):
    start_time = time.time()

    print(f"[+] Starting inference on: {upload_filename}")
    print("[*] Downloading input CSV from S3...")

    local_raw = "/tmp/input.csv"
    s3.download_file(S3_BUCKET, RAW_PREFIX + upload_filename, local_raw)

    print(f"    - File downloaded to {local_raw}")

    print("[*] Loading CSV...")
    df = pd.read_csv(local_raw)
    print(f"    - Loaded {len(df):,} rows.")

    MAX_ROWS = 5000
    if len(df) > MAX_ROWS:
        print(f"[!] Dataset has {len(df):,} rows — sampling down to {MAX_ROWS:,} rows...")
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    df = clean_and_engineer(df)

    result_list = []
    unique_cities = df["city"].unique()
    print(f"[+] Cities found: {unique_cities}\n")

    for city in unique_cities:
        print(f"[===] Processing city: {city} [===]")
        city_start = time.time()

        model = load_model(city)
        df_city = df[df["city"] == city].copy()

        X = df_city[get_features(df_city)]

        preds = np.expm1(model.predict(X))

        df_city["predicted_price"] = preds
        df_city["arbitrage_score"] = preds - df_city["price"]

        df_city["undervaluation_pct"] = ((df_city["predicted_price"] - df_city["price"]) 
                                 / df_city["predicted_price"]) * 100


        result_list.append(df_city)

        print(f"[✓] Finished {city} in {time.time() - city_start:.2f} sec\n")

    print("[*] Concatenating results...")
    final = pd.concat(result_list)

    # SAVE
    final.to_csv("/tmp/results.csv", index=False)
    s3.upload_file("/tmp/results.csv", S3_BUCKET, RESULT_FILE)

    print(f"[✓] Uploaded results to s3://{S3_BUCKET}/{RESULT_FILE}")
    print(f"\n[🏁] Total inference time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline_infer.py <csv_filename>")
        exit(1)

    filename = sys.argv[1]
    run_inference(filename)
