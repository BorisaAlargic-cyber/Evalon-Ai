import pandas as pd
import numpy as np


# ===== CONFIG =====
PRED_CSV = "EvalonAI_Results.csv"   # change path/name if needed


def main():
    df_preds = pd.read_csv(PRED_CSV)

    # Expect at minimum: city, price, predicted_price, constructedarea
    required_cols = {"city", "price", "predicted_price", "constructedarea"}
    missing = required_cols - set(df_preds.columns)
    if missing:
        print(f"[!] Missing required columns: {missing}")
        return

    # ---------- BASIC ERRORS ----------
    df_preds["abs_err"] = (df_preds["predicted_price"] - df_preds["price"]).abs()
    df_preds["pct_err"] = df_preds["abs_err"] / df_preds["price"].replace(0, np.nan) * 100

    print("\n=== Average price per city ===")
    print(df_preds.groupby("city")["price"].mean().rename("avg_price"))

    print("\n=== Absolute error by city (EUR) ===")
    print(df_preds.groupby("city")["abs_err"].agg(["count", "mean", "median"]))

    print("\n=== Percentage error by city (%) ===")
    print(df_preds.groupby("city")["pct_err"].agg(["count", "mean", "median"]))

    # ---------- TOP UNDER/OVER VALUATIONS ----------
    cols_show = [
        "city", "price", "predicted_price", "abs_err", "pct_err",
        "constructedarea", "roomnumber", "bathnumber",
        "neighborhood", "district"
    ]
    cols_show = [c for c in cols_show if c in df_preds.columns]

    print("\n=== TOP 10 HIGHEST PREDICTIONS (often 'undervalued' listings) ===")
    print(
        df_preds.sort_values("predicted_price", ascending=False)
                .head(10)[cols_show]
    )

    print("\n=== TOP 10 LOWEST PREDICTIONS (often 'overvalued' listings) ===")
    print(
        df_preds.sort_values("predicted_price", ascending=True)
                .head(10)[cols_show]
    )

    # ---------- WORST BY ABS ERROR PER CITY ----------
    for city in sorted(df_preds["city"].dropna().unique()):
        print(f"\n=== {city}: 10 worst by absolute error ===")
        print(
            df_preds[df_preds["city"] == city]
                .sort_values("abs_err", ascending=False)
                .head(10)[cols_show]
        )

    # ---------- ERROR BY SIZE SEGMENT ----------
    bins = [0, 40, 70, 100, 150, np.inf]
    labels = ["0–40", "40–70", "70–100", "100–150", "150+"]

    df_preds["size_bin"] = pd.cut(df_preds["constructedarea"], bins=bins, labels=labels)

    print("\n=== Percentage error by city and size segment ===")
    print(
        df_preds.groupby(["city", "size_bin"])["pct_err"]
                .agg(["count", "mean", "median"])
    )


if __name__ == "__main__":
    main()
