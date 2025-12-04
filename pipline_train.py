# pipeline_train.py
import pandas as pd
import numpy as np
import pickle
import boto3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# ------------ AWS CONFIG ------------
S3_BUCKET = "evalon-ai-team-1"
MODEL_PREFIX = "models/"
s3 = boto3.client("s3")

CITY_CENTERS = {
    "Barcelona": (41.3874, 2.1686),
    "Madrid":    (40.4168, -3.7038),
    "Valencia":  (39.4702, -0.3768),
}

# ----------- FEATURE ENGINEERING -----------
def calculate_distance(row):
    city = row.get("city")
    lat = row.get("latitude")
    lon = row.get("longitude")
    if city not in CITY_CENTERS or pd.isna(lat) or pd.isna(lon):
        return np.nan
    center_lat, center_lon = CITY_CENTERS[city]
    return ((lat - center_lat)**2 + (lon - center_lon)**2) ** 0.5

def clean_and_engineer(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    numeric_cols = ["price","constructedarea","bathnumber","roomnumber",
                    "latitude","longitude","builtyear"]
    for col in numeric_cols:
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
    features = [
        "constructedarea","bathnumber","roomnumber","distance_to_center",
        "price_m2","rooms_per_m2","bathrooms_per_room","builtyear",
        "propertytype","neighborhood","district"
    ]
    return [f for f in features if f in df.columns]

# ----------- MODEL BUILDER -----------
def build_model(X):
    categorical = X.select_dtypes(include="object").columns
    numeric = X.select_dtypes(include=["float64","int64"]).columns

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ])

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("pre", pre), ("model", model)])

# ----------- TRAIN + TEST SPLIT + SAVE MODEL -----------
def train_and_save_model(df, city):
    df_city = df[df["city"] == city].copy()

    X = df_city[get_features(df_city)]
    y = np.log1p(df_city["price"])  # log-transform

    # ---- TRAIN-TEST SPLIT ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = build_model(X_train)
    pipeline.fit(X_train, y_train)

    # ---- Evaluation ----
    preds = pipeline.predict(X_test)
    preds = np.expm1(preds)      # back-transform to €
    y_test_eur = np.expm1(y_test)

    r2 = r2_score(y_test_eur, preds)
    mae = mean_absolute_error(y_test_eur, preds)

    print(f"\n==============================")
    print(f"   {city} MODEL PERFORMANCE")
    print(f"==============================")
    print(f"R² Score:          {r2:.4f}")
    print(f"MAE (mean abs err): €{mae:,.2f}")
    print(f"==============================\n")

    # ---- SAVE MODEL ----
    filename = f"model_{city}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"[+] Saved model: {filename}")

    # Upload to S3
    s3.upload_file(filename, S3_BUCKET, MODEL_PREFIX + filename)
    print(f"[+] Uploaded to s3://{S3_BUCKET}/{MODEL_PREFIX}{filename}")

# ----------- MAIN -----------
def main():
    df = pd.read_csv("train_part.csv")
    df = clean_and_engineer(df)

    for city in sorted(df["city"].unique()):
        train_and_save_model(df, city)

if __name__ == "__main__":
    main()
