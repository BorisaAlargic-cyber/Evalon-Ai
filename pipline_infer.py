import pandas as pd
import numpy as np
import boto3
import pickle
import time
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

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

    # Distance to center
    df["distance_to_center"] = df.apply(calculate_distance, axis=1)
    df["distance_to_center"].fillna(df["distance_to_center"].median(), inplace=True)

    # Distance to metro (convert km → meters if necessary)
    if "distance_to_metro" in df.columns:
        if df["distance_to_metro"].max() < 100:  # looks like km
            df["distance_to_metro"] = df["distance_to_metro"] * 1000
    else:
        df["distance_to_metro"] = np.nan

    # Feature engineering
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
    start_time = time.time()

    local_raw = "/tmp/input.csv"
    s3.download_file(S3_BUCKET, RAW_PREFIX + upload_filename, local_raw)

    df = pd.read_csv(local_raw)

    # Reduce memory
    MAX_ROWS = 5000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    df = clean_and_engineer(df)

    result_list = []
    cities = df["city"].unique()

    # -----------------------------
    #    RUN MODEL FOR EACH CITY
    # -----------------------------
    for city in cities:
        df_city = df[df["city"] == city].copy()
        model = load_model(city)

        X = df_city[get_features(df_city)]
        preds = np.expm1(model.predict(X))

        df_city["predicted_price"] = preds
        df_city["arbitrage_score"] = preds - df_city["price"]
        df_city["undervaluation_pct"] = (df_city["arbitrage_score"] / df_city["price"]) * 100

        # Save results
        result_list.append(df_city)

        # -----------------------------
        #      INFERENCE PERFORMANCE
        # -----------------------------
        y_true = df_city["price"]
        y_pred = preds

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print("\n==============================")
        print(f"   {city} INFERENCE PERFORMANCE")
        print("==============================")
        print(f"R² Score:          {r2:.4f}")
        print(f"MAE:               €{mae:,.2f}")
        print(f"Median Abs Err:    €{medae:,.2f}")
        print(f"MAPE:              {mape:.2f}%")
        print("==============================\n")

    # CONCAT ALL RESULTS
    final = pd.concat(result_list)

    # TOP 50 PER CITY
    final_top = (
        final.sort_values("arbitrage_score", ascending=False)
             .groupby("city")
             .head(50)
    )

    final_top.to_csv("/tmp/results.csv", index=False)
    s3.upload_file("/tmp/results.csv", S3_BUCKET, RESULT_FILE)

    print(f"[✓] Uploaded results to s3://{S3_BUCKET}/{RESULT_FILE}")
    print(f"[🏁] Total inference time: {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 pipline_infer.py <csv_filename>")
        exit(1)

    filename = sys.argv[1]
    run_inference(filename)
