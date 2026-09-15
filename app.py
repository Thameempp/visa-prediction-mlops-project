import sys

from fastapi import FastAPI

from us_visa.exception import UsVisaException
from us_visa.pipeline.training_pipeline import TrainPipeline

app = FastAPI(title="US Visa Prediction API", version="0.0.1")


@app.get("/")
def read_root() -> dict:
    return {"message": "US Visa project API is running"}


@app.post("/train")
def train_pipeline() -> dict:
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return {"message": "Training pipeline completed successfully"}
    except Exception as e:
        raise UsVisaException(e, sys) from e
