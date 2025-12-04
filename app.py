# app.py
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import boto3
import subprocess

S3_BUCKET = "evalon-ai-team-1"
RAW_PREFIX = "raw/"
RESULT_PREFIX = "results/"
RESULT_FILE = RESULT_PREFIX + "rf_top50_each_city.csv"

s3 = boto3.client("s3")

app = Flask(__name__)
app.secret_key = "supersecretkey"

# =====================================================
# UPLOAD + AUTO RUN INFERENCE
# =====================================================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        filename = secure_filename(f.filename)

        # Save local temp
        local_path = f"/tmp/{filename}"
        f.save(local_path)

        # Upload raw CSV to S3
        s3.upload_file(local_path, S3_BUCKET, RAW_PREFIX + filename)
        flash("File uploaded to S3. Running inference...", "info")

        # RUN PIPELINE
        try:
            cmd = ["python3", "pipline_infer.py", filename]
            subprocess.run(cmd, check=True)
            flash("Inference complete! Dashboard updated.", "success")
        except Exception as e:
            flash(f"Inference FAILED: {e}", "danger")

        return redirect(url_for("dashboard"))

    return render_template("upload.html")

# =====================================================
# DOWNLOAD OUTPUT CSV
# =====================================================
@app.route("/download")
def download():
    """Download the result CSV file stored in S3."""
    local_path = "/tmp/download_results.csv"

    try:
        s3.download_file(S3_BUCKET, RESULT_FILE, local_path)
    except Exception as e:
        return f"Error downloading file: {e}", 500

    return send_file(
        local_path,
        as_attachment=True,
        download_name="EvalonAI_Results.csv"
    )

# =====================================================
# DASHBOARD
# =====================================================
@app.route("/", methods=["GET"])
def dashboard():
    local_out = "/tmp/out.csv"

    # Download inference results from S3
    try:
        s3.download_file(S3_BUCKET, RESULT_FILE, local_out)
    except:
        return render_template(
            "index.html",
            results=[],
            cities=[],
            total_results=0
        )

    df = pd.read_csv(local_out)

    # Rename-fields for UI
    df = df.rename(columns={
        "distance_to_center": "dist_center_m",
        "distance_to_metro": "dist_metro_m",
        "assetId": "assetid"
    })

    # Arbitrage score
    df["arbitrage_score"] = df["predicted_price"] - df["price"]

    # Undervaluation %
    df["undervaluation_pct"] = (
        (df["predicted_price"] - df["price"]) / df["price"] * 100
    ).fillna(0)

    # --------- FILTERS ----------
    selected_city = request.args.get("city", "All")
    selected_rooms = int(request.args.get("rooms", 1))
    selected_baths = int(request.args.get("bathrooms", 1))
    selected_parking = request.args.get("parking", "All")
    sort_option = request.args.get("sort", "none")

    filtered = df.copy()

    # City
    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]

    # Rooms / Baths
    filtered = filtered[
        (filtered["roomnumber"] >= selected_rooms) &
        (filtered["bathnumber"] >= selected_baths)
    ]

    # Parking
    if selected_parking != "All":
        filtered = filtered[
            filtered["hasparkingspace"] == (selected_parking == "Yes")
        ]

    # --------- SORTING ---------
    if sort_option == "arbitrage_desc":
        filtered = filtered.sort_values(by="arbitrage_score", ascending=False)

    if sort_option == "undervaluation_desc":
        filtered = filtered.sort_values(by="undervaluation_pct", ascending=False)

    # Backend → UI rename
    filtered = filtered.rename(columns={
        "roomnumber": "rooms",
        "bathnumber": "bathrooms",
        "hasparkingspace": "has_parking",
    })

    results = filtered.to_dict(orient="records")

    return render_template(
        "index.html",
        cities=sorted(df["city"].unique()),
        selected_city=selected_city,
        selected_rooms=selected_rooms,
        selected_baths=selected_baths,
        selected_parking=selected_parking,
        sort_option=sort_option,
        results=results,
        total_results=len(df)
    )

# ------------------------------
# Performance Dashboard
# ------------------------------
@app.route("/performance")
def performance():
    local_out = "/tmp/out.csv"

    # Try to download the latest results
    try:
        s3.download_file(S3_BUCKET, RESULT_FILE, local_out)
    except Exception as e:
        return f"Results file missing: {e}"

    df = pd.read_csv(local_out)

    # --- Compute Metrics ---
    from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

    y_true = df["price"]
    y_pred = df["predicted_price"]

    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = (abs((y_true - y_pred) / y_true).mean()) * 100
    r2 = r2_score(y_true, y_pred)

    # --- Per-City Breakdown ---
    city_stats = []
    for city in df["city"].unique():
        sub = df[df["city"] == city]
        c_y_true = sub["price"]
        c_y_pred = sub["predicted_price"]

        c_mae = mean_absolute_error(c_y_true, c_y_pred)
        c_medae = median_absolute_error(c_y_true, c_y_pred)
        c_mape = (abs((c_y_true - c_y_pred) / c_y_true).mean()) * 100
        c_r2 = r2_score(c_y_true, c_y_pred)

        city_stats.append({
            "city": city,
            "mae": round(c_mae, 2),
            "medae": round(c_medae, 2),
            "mape": round(c_mape, 2),
            "r2": round(c_r2, 3),
        })

    return render_template(
        "performance.html",
        mae=round(mae, 2),
        medae=round(medae, 2),
        mape=round(mape, 2),
        r2=round(r2, 3),
        city_stats=city_stats,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
