# Evalon AI - Real Estate Valuation & Sourcing Platform

**Final Project: Cloud Computing & Machine Learning Implementation**

## Executive Summary

**Evalon AI** is a B2B SaaS platform that turns raw real estate data into actionable investment intelligence.

We address critical inefficiencies in deal sourcing by using a **Random Forest model** to automate preliminary valuations. In this first version, analysts simply upload a CSV file with a list of properties, and our system calculates a specific **'Undervaluation Score'**. This allows investment teams to filter thousands of listings instantly and prioritize only the properties worth a deeper manual analysis.

We used an Idealista’s database with **~190k properties** in Barcelona, Madrid, and Valencia. The application is built with **Flask** and fully integrated with **AWS services (S3, EC2, EBS, IAM)** for scalable, cloud-native data management.

## The Problem Statement

Real estate investment analysts currently face three major bottlenecks:

1.  **Manual Sourcing (Efficiency):** Analysts spend hundreds of hours manually browsing portals like Idealista, analyzing listings one by one. This process is unscalable.
2.  **Valuation Speed:** Investment teams need rapid answers. Determining "Which of these 10,000 listings are actually undervalued?" is currently a slow, manual calculation.
3.  **Resource Allocation:** Analysts waste too much time on rough, back-of-the-envelope checks for bad deals, rather than focusing their energy on deep-diving into the few viable opportunities.

## Our Solution

We built a cloud-deployed architecture that automates the initial screening pipeline, acting as a "Deal Co-Pilot" for analysts:

* **Automated Ingestion:** The system accepts raw CSV uploads of market data, processing thousands of rows instantly.
* **Preliminary Screening Engine:** Instead of replacing human valuation, our Random Forest model (trained on **~95,000 listings**) acts as a first-pass filter. It identifies statistical anomalies where the listed price is significantly lower than the predicted market capability.
* **Prioritization for "Deep Dives":** The dashboard ranks listings by their potential upside. This allows analysts to ignore the potentially overpriced assets and focus their manual "deep dive" efforts only on the high-potential opportunities.

### Dataset & Features

Our model was trained on a robust dataset containing 29 raw features, engineered down to 20 key predictors, including:

* **Structural:** Area ($m^2$), Rooms, Bathrooms, Year Built, etc.
* **Location:** Neighborhood (One-Hot Encoded), Distance to Center.
* **Amenities:** Elevator, Terrace, A/C, etc.

## System Architecture & Cloud Data Flow

The project utilizes a realistic "PropTech" architecture pattern, deliberately separating **Offline Training** (resource-intensive, infrequent) from **Online Inference** (low-latency, high-availability).

### 1. Architectural Overview & Data Flow

The system is designed as a monolithic web application hosted on cloud infrastructure, ensuring persistence and scalability. The data flow follows a strict lifecycle to ensure data integrity and traceability:

1.  **Ingestion Layer:** The user interacts with the Flask web interface hosted on **AWS EC2**. Upon uploading a CSV dataset, the system uses **Boto3** to immediately duplicate the raw file into an **AWS S3** bucket (`/raw-uploads/`). This guarantees that user data is backed up before processing begins.
2.  **Inference Layer:** The EC2 instance retrieves the pre-trained **Random Forest model** from its local **AWS EBS** volume. By keeping the model artifact on local block storage rather than downloading it from S3 for every request, we achieve sub-second model loading times. The application then processes the CSV in memory, running inference on every row to generate fair value predictions.
3.  **Archival Layer:** Once the "Undervaluation Scores" are computed, the results are compiled into an Investment Report. This report is automatically uploaded back to **AWS S3** (`/reports/`), creating a permanent audit trail of all analyses performed by the firm.
4.  **Presentation Layer:** Finally, the processed results are rendered in the user's browser, displaying a ranked list of top investment opportunities.

### 2. AWS Component Configuration

We utilized the following AWS services to build a robust, production-grade environment:

#### A. AWS EC2 (Elastic Compute Cloud)
* **Function:** Acts as the primary compute resource hosting the Flask application server and the Scikit-Learn inference engine.
* **Justification:** We selected EC2 over serverless options (like AWS Lambda) to maintain a persistent state. This is critical for loading the large Machine Learning model into memory once at startup, rather than reloading it for every request. It also allows for long-running batch processing of large datasets without encountering the execution time limits common in serverless environments.

#### B. AWS EBS (Elastic Block Store)
* **Function:** Provides persistent block-level storage attached to the EC2 instance, since the default EC2 storage was not sufficient due to the amount of data we had.
* **Usage:**
    * **Codebase:** Houses the application source code (`app.py`, templates).
    * **Model Artifact:** Stores the serialized `barcelona_housing_model.pkl`. Hosting the model here ensures low-latency access and rapid I/O performance during application startup.

#### C. AWS S3 (Simple Storage Service)
* **Function:** Serves as the centralized object storage repository ("Data Lake").
* **Usage:**
    * **Decoupling:** By storing data in S3 rather than on the EC2 instance itself, we decouple compute from storage. This allows us to terminate or replace the EC2 instance without losing any historical client data.

#### D. AWS IAM (Identity and Access Management)
* **Function:** Manages access control and permissions.
* **Implementation:** We implemented a "Least Privilege" security model by attaching a specific IAM Role to the EC2 instance. This role grants permission only for `s3:PutObject` and `s3:GetObject` operations within our specific bucket. This eliminates the security risk of storing hardcoded AWS Access Keys within the application source code.

## Detailed Codebase Walkthrough

This section details the technical implementation of each file in the repository.

### 1. app.py
This is the core entry point of the application, built using the **Flask** microframework. It orchestrates the interaction between the user, the Machine Learning model, and AWS services.

* **Key Functions:**
    * `configure_aws()`: Fetches secure credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) from environment variables.
    * `load_model()`: Reads `barcelona_housing_model.pkl` from the EBS volume into memory when the server starts.
* **Route Logic:**
    * **POST `/predict`**: Handles validation, S3 upload of raw data, preprocessing, inference, and S3 upload of the final report.

### 2. pipline_train.py (Offline Training)
This script handles the machine learning development lifecycle.

* **Functionality:** It splits the dataset into **95k for training/validation** and **95k for holdout testing**. It then performs feature engineering (One-Hot Encoding, Log Transformation), trains the Random Forest model, and serializes the final pipeline into a `.pkl` file.
* **Usage:** This is run locally or on a training instance to generate the model artifact.

### 3. pipline_infer.py (Inference Logic)
This module encapsulates the prediction logic used by the Flask application.

* **Purpose:** It ensures modularity by separating the web server logic (`app.py`) from the core data science logic. It contains functions to preprocess new incoming CSV data to match the format expected by the trained model.

### 4. train_part.csv (Sample Data)
A subset of the larger ~95k listing dataset.
* **Purpose:** This is used to train the `pipline_train.py` script.

### 5. `templates/` (The Frontend Interface)
We used the **Jinja2** templating engine to generate dynamic HTML and handle template inheritance.

* **`base.html`:** The master layout file. It defines the common structure (navigation bar, footer, CSS imports) that applies to all pages, ensuring a consistent UI.
* **`index.html`:** The landing page. Extends `base.html` to provide the main entry point and instructions for the user.
* **`upload.html`:** The file upload interface. It contains the form configured with `enctype="multipart/form-data"` to handle CSV transmission to the backend.
* **`performance.html`:** The results dashboard. It dynamically renders the "Investment Report" table using data passed from the backend, highlighting top opportunities.

### 6. `uploads/` (Temporary Buffer)
A local directory on the EC2 instance used as a temporary staging area.
* **Process:** When Flask receives a file, it saves it here first. Boto3 then reads from this location to upload to S3.

### 7. requirements.txt (Dependency Management)
Ensures reproducibility of the environment.
* `Flask`, `boto3`, `pandas`, `scikit-learn`: Defines the exact versions of libraries required to run the application on EC2.

## Future Work & Roadmap

### Short-Term Outlook (Product & Engineering)
* **User Authentication:** Implement private logins per user to secure proprietary data.
* **Saved Filters:** Allow analysts to save search criteria (e.g., "Yield > 5% in Eixample").
* **Scheduled Pipelines:** Convert the manual EC2 upload trigger into a fully automated scheduled job using AWS Lambda or Cron.
* **Monitoring:** Integrate **AWS CloudWatch** for logging application health and error tracking.

### Long-Term Vision (Platform & Ecosystem)
* **Live Data Sourcing:** Integrate directly with the **Idealista API** for real-time deal flow, removing the need for CSV uploads.
* **"Deal Co-Pilot":** Enhance the ML model to suggest target acquisition prices and estimated renovation budgets.
* **Pan-European Expansion:** Retrain models on data from Paris, Berlin, and Lisbon.
* **External API:** Expose our "Fair Value Score" via a REST API for banks and wealth managers to embed in their own tools.

## Team Contributors - Team 1
- Jerico Agdan
- Borisa Alargic
- Sabeena Awan
- Lorenzo Costa
- Sara Saleem
