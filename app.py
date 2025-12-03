# app.py
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import boto3

S3_BUCKET = "evalon-ai-team-1"
RAW_PREFIX = "raw/"
RESULT_PREFIX = "results/"
RESULT_FILE = RESULT_PREFIX + "rf_top50_each_city.csv"

s3 = boto3.client("s3")

app = Flask(__name__)
app.secret_key = "supersecretkey"


# ------------------------------
# Upload to S3
# ------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        filename = secure_filename(f.filename)
        local_path = f"/tmp/{filename}"
        f.save(local_path)

        # Upload raw file to S3
        s3.upload_file(local_path, S3_BUCKET, RAW_PREFIX + filename)

        flash("Uploaded to S3! Now run inference.", "success")
        return redirect(url_for("dashboard"))

    return render_template("upload.html")


# ------------------------------
# Dashboard (MAIN UI)
# ------------------------------
@app.route("/", methods=["GET"])
def dashboard():

    local_out = "/tmp/out.csv"

    # Try downloading the result CSV from S3
    try:
        s3.download_file(S3_BUCKET, RESULT_FILE, local_out)
    except:
        return render_template("index.html",
                               results=[],
                               cities=[],
                               neighbourhoods=[],
                               total_results=0)

    df = pd.read_csv(local_out)

    # Visual rename for UI
    df = df.rename(columns={
        "distance_to_center": "dist_center_km",
        "distance_to_metro": "dist_metro_m",
        "assetId": "assetid"
    })

    # Ensure arbitrage column exists
    if "arbitrage_score" not in df.columns:
        df["arbitrage_score"] = df["predicted_price"] - df["price"]

    # Get filters
    selected_city = request.args.get("city", "All")
    selected_rooms = int(request.args.get("rooms", 1))
    selected_baths = int(request.args.get("bathrooms", 1))
    selected_parking = request.args.get("parking", "All")

    # Start filtering
    filtered = df.copy()

    # City filter
    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]

    # Rooms + Bathrooms filter (CSV fields!)
    filtered = filtered[
        (filtered["roomnumber"] >= selected_rooms) &
        (filtered["bathnumber"] >= selected_baths)
    ]

    # Parking filter
    if selected_parking != "All":
        filtered = filtered[
            filtered["hasparkingspace"] == (selected_parking == "Yes")
        ]

    # 🟢 Rename backend → frontend fields
    filtered = filtered.rename(columns={
        "roomnumber": "rooms",
        "bathnumber": "bathrooms",
        "hasparkingspace": "has_parking",
    })

    results = filtered.to_dict(orient="records")

    return render_template("index.html",
                           cities=sorted(df["city"].unique()),
                           neighbourhoods=[],
                           selected_city=selected_city,
                           selected_rooms=selected_rooms,
                           selected_baths=selected_baths,
                           selected_parking=selected_parking,
                           results=results,
                           total_results=len(df))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
