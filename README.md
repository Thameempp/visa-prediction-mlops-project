# US Visa Prediction Project

This project builds an end-to-end machine learning pipeline for predicting US visa case status, such as whether an application is likely to be `Certified` or `Denied`.

The pipeline uses applicant, employer, job, wage, and region details from the visa dataset, then trains a classification model that can be saved and reused for prediction. The source CSV is available at `Notebook/Visadataset.csv`.

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
├── Notebook/
│   ├── Visadataset.csv            # Source visa dataset
│   └── *.ipynb                    # EDA and experiment notebooks
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
├── scripts/
│   └── setup.sh                   # Recreates ignored local setup files/folders
├── setup.py
└── Dockerfile
```

## Setup

Run the setup script:

```bash
bash scripts/setup.sh
```

This creates the ignored runtime folders, creates `.env` from `.env.example` if needed, creates a local `.venv`, and installs the Python dependencies.

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Update `.env` with your MongoDB connection string:

```text
DB_URL=your_mongodb_connection_string
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
- Ignored runtime folders and `.env` are recreated by `bash scripts/setup.sh`.
- The dataset file is stored at `Notebook/Visadataset.csv`.
- The training pipeline currently expects MongoDB to contain the `US_VISA.visa_data` collection.
- The target column is `case_status`.
