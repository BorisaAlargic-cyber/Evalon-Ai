
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

CITY_CENTERS = {
    "Barcelona": (41.3874, 2.1686),
    "Madrid":    (40.4168, -3.7038),
    "Valencia":  (39.4702, -0.3768),
}

def calculate_distance(row):
    city = row.get("city", None)
    lat = row.get("latitude", None)
    lon = row.get("longitude", None)
    if city not in CITY_CENTERS or pd.isna(lat) or pd.isna(lon):
        return np.nan
    center_lat, center_lon = CITY_CENTERS[city]
    return ((lat - center_lat)**2 + (lon - center_lon)**2) ** 0.5

def clean_and_engineer(df):
    df = df.copy()
    drop_cols = ['id','url','operation','externalReference','thumbnail','newDevelopment']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df.columns = [c.lower() for c in df.columns]
    numeric_candidates = ["price","constructedarea","bathnumber","roomnumber","unitprice",
                          "level","elevation","caddwellingcount","builtyear","latitude","longitude"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "price" in df.columns:
        df = df[(df["price"] < 2_500_000) & (df["price"] > 30_000)]
    if "constructedarea" in df.columns:
        df = df[df["constructedarea"] < 400]

    bool_cols = ["haslift","hasparking","hasairconditioning","hasterrace",
                 "hasseaview","hasheating","hasgarden","hasbalcony"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    df["distance_to_center"] = df.apply(calculate_distance, axis=1)
    df["distance_to_center"] = df["distance_to_center"].fillna(df["distance_to_center"].median())

    if "price" in df.columns and "constructedarea" in df.columns:
        df["price_m2"] = df["price"] / df["constructedarea"]
    if "roomnumber" in df.columns and "constructedarea" in df.columns:
        df["rooms_per_m2"] = df["roomnumber"] / df["constructedarea"]
    if "bathnumber" in df.columns and "roomnumber" in df.columns:
        df["bathrooms_per_room"] = df["bathnumber"] / df["roomnumber"].replace(0, np.nan)

    for col in df.columns:
        if df[col].dtype in ["float64","int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

def get_features(df):
    feature_list = ["constructedarea","bathnumber","roomnumber","level","elevation",
                    "distance_to_center","price_m2","rooms_per_m2","bathrooms_per_room",
                    "haslift","hasparking","hasairconditioning","hasterrace",
                    "hasheating","hasgarden","hasbalcony","builtyear",
                    "propertytype","neighborhood","district"]
    return [f for f in feature_list if f in df.columns]

def build_preprocessor_and_model(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline([("pre", pre), ("model", model)])
    return pipeline

def naive_baseline_mae(df):
    city_mean = df["price"].mean()
    preds = np.ones(len(df)) * city_mean
    return mean_absolute_error(df["price"], preds)

def run_kfold_rf(df, n_splits=5, city_name=None):
    y = np.log1p(df["price"])
    X = df[get_features(df)]
    pipeline = build_preprocessor_and_model(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []

    print(f"\n=== KFold RF for {city_name} ===")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X),1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        preds = np.expm1(pipeline.predict(X_test))
        true_vals = np.expm1(y_test)
        mae = mean_absolute_error(true_vals, preds)
        fold_maes.append(mae)
        print(f"Fold {fold} MAE ({city_name}): {mae:,.2f} €")

    print(f"RF Average MAE ({city_name}): {np.mean(fold_maes):,.2f} €")
    baseline = naive_baseline_mae(df)
    print(f"Naive baseline MAE ({city_name}): {baseline:,.2f} €")
    return np.mean(fold_maes), baseline

def undervaluation_top50_each_city(df, n_splits=5, min_pct=0.10, top_n=50):
    df = df.copy()
    all_top = []

    for city in sorted(df["city"].unique()):
        df_city = df[df["city"] == city].copy()
        if len(df_city) < n_splits:
            print(f"Skipping {city}: not enough rows.")
            continue

        print(f"\n--- Undervaluation for {city} ---")
        y = np.log1p(df_city["price"])
        X = df_city[get_features(df_city)]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(df_city))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X),1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            pipeline = build_preprocessor_and_model(X)
            pipeline.fit(X_train, y_train)
            oof_preds[val_idx] = np.expm1(pipeline.predict(X_val))

        df_city["predicted_price_oof"] = oof_preds
        df_city["undervaluation_abs"] = df_city["predicted_price_oof"] - df_city["price"]
        df_city["undervaluation_pct"] = df_city["undervaluation_abs"] / df_city["predicted_price_oof"]

        undervalued = df_city[df_city["undervaluation_pct"] >= min_pct]
        top50 = undervalued.sort_values("undervaluation_abs", ascending=False).head(top_n)
        all_top.append(top50)
        print(f"Top undervalued properties in {city}: {len(top50)}")

    final_top = pd.concat(all_top, axis=0)
    final_top.to_csv("rf_top50_each_city.csv", index=False)
    print("\nSaved rf_top50_each_city.csv")

def run_pipeline(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print("Cleaning & feature engineering...")
    df_clean = clean_and_engineer(df)

    for city in sorted(df_clean["city"].unique()):
        df_city = df_clean[df_clean["city"] == city]
        run_kfold_rf(df_city, n_splits=5, city_name=city)

    undervaluation_top50_each_city(df_clean, n_splits=5)

if __name__ == "__main__":
    run_pipeline("Spain_RealEstate_Merged.csv")
