# US Visa Prediction Project

This project builds an end-to-end machine learning pipeline for predicting US visa case status, such as whether an application is likely to be `Certified` or `Denied`.

The pipeline uses applicant, employer, job, wage, and region details from the visa dataset, then trains a classification model that can be saved and reused for prediction.

## Project Features

- Ingests visa data from MongoDB
- Splits data into train and test sets
- Validates dataset columns against `config/schema.yaml`
- Performs feature engineering and preprocessing
- Trains a machine learning classifier
- Saves trained models and preprocessing artifacts
- Provides a FastAPI app with a training endpoint
- Includes Docker support

## Project Structure

```text
.
├── app.py                         # FastAPI application
├── config/
│   ├── model.yaml                 # Model search/config placeholder
│   └── schema.yaml                # Dataset schema
├── Notebook/                      # EDA and experiment notebooks
├── src/us_visa/
│   ├── components/                # ML pipeline components
│   ├── configuration/             # MongoDB connection
│   ├── constants/                 # Project constants
│   ├── data_access/               # Data export from MongoDB
│   ├── entity/                    # Config and artifact dataclasses
│   ├── exception/                 # Custom exception handling
│   ├── logger/                    # Logging setup
│   ├── pipeline/                  # Training and prediction pipelines
│   └── utils/                     # Reusable utility functions
├── requirements.txt
├── setup.py
└── Dockerfile
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your MongoDB connection string:

```bash
export DB_URL="your_mongodb_connection_string"
```

## Run Training

Run the training pipeline directly:

```bash
python demo.py
```

Or start the FastAPI app:

```bash
uvicorn app:app --reload
```

Then call:

```bash
POST /train
```

## Docker

Build the image:

```bash
docker build -t us-visa-prediction .
```

Run the container:

```bash
docker run -p 8000:8000 -e DB_URL="your_mongodb_connection_string" us-visa-prediction
```

## Notes

- Generated training outputs are ignored by Git: `artifact/`, `artifacts/`, `saved_models/`, and local data folders.
- The project expects MongoDB to contain the `US_VISA.visa_data` collection.
- The target column is `case_status`.
