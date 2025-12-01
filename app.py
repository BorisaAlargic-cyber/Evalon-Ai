import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import NoCredentialsError

# -------------------------------
# AWS CONFIG
# -------------------------------
S3_BUCKET = "your-bucket-name"         # ← CHANGE THIS
RAW_PREFIX = "raw/"
RESULT_PREFIX = "results/"
RESULT_FILE = RESULT_PREFIX + "rf_top50_each_city.csv"

# boto3 client (credentials come from IAM role on EC2)
s3 = boto3.client("s3")

app = Flask(__name__)
app.secret_key = "supersecretkey"


# --------------------------------------
# Helper: Upload File to S3
# --------------------------------------
def upload_to_s3(local_path, bucket_path):
    try:
        s3.upload_file(local_path, S3_BUCKET, bucket_path)
        return True
    except Exception as e:
        print("Upload error:", e)
        return False


# --------------------------------------
# Helper: Download File from S3
# --------------------------------------
def download_from_s3(bucket_path, local_path):
    try:
        s3.download_file(S3_BUCKET, bucket_path, local_path)
        return True
    except Exception as e:
        print("Download error:", e)
        return False


# --------------------------------------
# UPLOAD PAGE
# --------------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        local_tmp = f"/tmp/{filename}"
        file.save(local_tmp)

        # Upload to S3
        s3_key = RAW_PREFIX + filename
        success = upload_to_s3(local_tmp, s3_key)

        if success:
            flash("File uploaded to S3 successfully!", "success")
        else:
            flash("Upload to S3 failed.", "error")

        return redirect(url_for("dashboard"))

    return render_template("upload.html")


# --------------------------------------
# DASHBOARD PAGE
# --------------------------------------
@app.route("/", methods=["GET"])
def dashboard():

    local_result = "/tmp/final_results.csv"

    # Attempt to download the ML output from S3
    if not download_from_s3(RESULT_FILE, local_result):
        return render_template("index.html",
                               cities=[],
                               assetid=[],
                               results=[],
                               total_results=0)

    # Load results
    df = pd.read_csv(local_result)

    # Normalize for template compatibility
    rename_map = {
        "predicted_price_oof": "predicted_price",
        "distance_to_center": "dist_center_km",
        "distance_to_metro": "dist_metro_m",
        "roomnumber": "rooms",
        "bathnumber": "bathrooms",
        "hasparkingspace": "has_parking",
    }

    df = df.rename(columns=rename_map)

    df["arbitrage_score"] = df["predicted_price"] - df["price"]

    # ------------- Filters -------------
    selected_city = request.args.get("city", "All")
    selected_rooms = int(request.args.get("rooms", 1))
    selected_baths = int(request.args.get("bathrooms", 1))
    selected_parking = request.args.get("parking", "All")

    filtered = df.copy()

    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]

    filtered = filtered[
        (filtered["rooms"] >= selected_rooms) &
        (filtered["bathrooms"] >= selected_baths)
    ]

    if selected_parking != "All":
        filtered = filtered[filtered["has_parking"] == (selected_parking == "Yes")]

    results = filtered.to_dict(orient="records")

    return render_template(
        "index.html",
        cities=sorted(df["city"].unique()),
        assetid=[],   # Removed neighborhood: unused
        selected_city=selected_city,
        selected_hood="All",
        selected_rooms=selected_rooms,
        selected_baths=selected_baths,
        selected_parking=selected_parking,
        results=results,
        total_results=len(df)
    )


# --------------------------------------
# RUN
# --------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
