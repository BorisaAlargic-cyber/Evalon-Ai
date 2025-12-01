# pipeline_infer.py
import pandas as pd
import numpy as np
import boto3
import pickle

S3_BUCKET = "evalon-ai-team1"
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

def calculate_distance(row):
    city = row.get("city")
    lat = row.get("latitude")
    lon = row.get("longitude")
    if city not in CITY_CENTERS or pd.isna(lat) or pd.isna(lon):
        return np.nan
    c_lat, c_lon = CITY_CENTERS[city]
    return ((lat - c_lat)**2 + (lon - c_lon)**2)**0.5

def clean_and_engineer(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    num_cols = ["price","constructedarea","bathnumber","roomnumber",
                "latitude","longitude","builtyear"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["distance_to_center"] = df.apply(calculate_distance, axis=1)
    df["distance_to_center"].fillna(df["distance_to_center"].median(), inplace=True)

    df["price_m2"] = df["price"] / df["constructedarea"]
    df["rooms_per_m2"] = df["roomnumber"] / df["constructedarea"]
    df["bathrooms_per_room"] = df["bathnumber"] / df["roomnumber"].replace(0, np.nan)

    df = df.fillna(df.median(numeric_only=True)).fillna("Unknown")
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
    s3.download_file(S3_BUCKET, s3_key, local_path)
    return pickle.load(open(local_path, "rb"))

def run_inference(upload_filename):
    # Download RAW CSV
    local_raw = "/tmp/input.csv"
    s3.download_file(S3_BUCKET, RAW_PREFIX + upload_filename, local_raw)

    df = pd.read_csv(local_raw)
    df = clean_and_engineer(df)

    result_list = []

    for city in df["city"].unique():
        model = load_model(city)
        df_city = df[df["city"] == city].copy()

        X = df_city[get_features(df_city)]
        preds = np.expm1(model.predict(X))

        df_city["predicted_price"] = preds
        df_city["arbitrage_score"] = preds - df_city["price"]

        result_list.append(df_city)

    final = pd.concat(result_list)
    final.to_csv("/tmp/results.csv", index=False)

    s3.upload_file("/tmp/results.csv", S3_BUCKET, RESULT_FILE)
    print("[+] Uploaded results to S3")

if __name__ == "__main__":
    run_inference("YOUR_UPLOADED_FILE.csv")
