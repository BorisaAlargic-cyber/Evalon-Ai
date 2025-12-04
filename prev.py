import pandas as pd
import numpy as np
import pickle
import boto3

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


S3_BUCKET = "evalon-ai-team-1"
MODEL_PREFIX = "models/"
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
    center_lat, center_lon = CITY_CENTERS[city]
    return ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # numeric casting
    numeric_cols = [
        "price", "constructedarea", "bathnumber", "roomnumber",
        "latitude", "longitude", "builtyear"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # feature engineering
    df["distance_to_center"] = df.apply(calculate_distance, axis=1)

    df["price_m2"] = df["price"] / df["constructedarea"]
    df["rooms_per_m2"] = df["roomnumber"] / df["constructedarea"]
    df["bathrooms_per_room"] = df["bathnumber"] / df["roomnumber"].replace(0, np.nan)

    # basic target cleaning: drop rows without price
    df = df[~df["price"].isna()].copy()

    # simple categorical cleaning
    for col in ["propertytype", "neighborhood", "district", "city"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def get_features(df: pd.DataFrame):
    features = [
        "constructedarea", "bathnumber", "roomnumber", "distance_to_center",
        "price_m2", "rooms_per_m2", "bathrooms_per_room", "builtyear",
        "propertytype", "neighborhood", "district"
    ]
    return [f for f in features if f in df.columns]


def build_model(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include="object").columns
    numeric = X.select_dtypes(include=["float64", "int64", "int32"]).columns

    # explicit imputers inside the pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,          # let trees grow deeper; leaf size prevents overfit
        min_samples_leaf=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("pre", pre), ("model", model)])


def evaluate_model(pipeline: Pipeline, X_test, y_test_log):
    # predictions
    preds_log = pipeline.predict(X_test)
    preds = np.expm1(preds_log)
    y_test = np.expm1(y_test_log)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return r2, mae


def train_and_save_model(df: pd.DataFrame, city: str):
    df_city = df[df["city"] == city].copy()
    if df_city.shape[0] < 200:
        print(f"[!] Skipping {city}: too few rows ({df_city.shape[0]}).")
        return

    X = df_city[get_features(df_city)]
    y_log = np.log1p(df_city["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )

    pipeline = build_model(X_train)

    # optional cross‑validation on the training set (log‑space R²)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=3, scoring="r2", n_jobs=-1
    )

    pipeline.fit(X_train, y_train)

    r2, mae = evaluate_model(pipeline, X_test, y_test)

    print("\n==============================")
    print(f"   {city} MODEL PERFORMANCE")
    print("==============================")
    print(f"CV R² (log-price):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test R² (price €):  {r2:.4f}")
    print(f"Test MAE:           €{mae:,.2f}")
    print("==============================\n")

    filename = f"model_{city}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[+] Saved model: {filename}")

    s3.upload_file(filename, S3_BUCKET, MODEL_PREFIX + filename)
    print(f"[+] Uploaded to s3://{S3_BUCKET}/{MODEL_PREFIX}{filename}")


def main():
    df = pd.read_csv("train_part.csv")
    df = clean_and_engineer(df)

    for city in sorted(df["city"].dropna().unique()):
        train_and_save_model(df, city)


if __name__ == "__main__":
    main()
