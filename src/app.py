from src.scheduler import Scheduler, SchedulerConfig
from src.models import Job, JobCreate, JobUpdate
from fastapi import FastAPI, HTTPException, status
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from typing import List

# Load environment variables
load_dotenv()

app = FastAPI()
config = SchedulerConfig(
    mongodb_uri=SecretStr(os.getenv("MONGODB_URI", "")),
    telegram_bot_token=SecretStr(os.getenv("TELEGRAM_BOT_TOKEN", "")),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
)
scheduler = Scheduler(config)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/jobs", response_model=Job, status_code=status.HTTP_201_CREATED)
async def add_job(job: JobCreate):
    """Create a new job."""
    try:
        return await scheduler.add_job(job)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding job: {str(e)}",
        )


@app.get("/jobs", response_model=List[Job])
async def get_jobs():
    """Get all jobs."""
    return await scheduler.get_jobs()


@app.get("/jobs/{job_key}", response_model=Job)
async def get_job(job_key: str):
    """Get a specific job by key."""
    job = await scheduler.get_job(job_key)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with key '{job_key}' not found",
        )
    return job


@app.put("/jobs/{job_key}", response_model=Job)
async def update_job(job_key: str, job_update: JobUpdate):
    """Update a job."""
    job = await scheduler.update_job(job_key, job_update)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with key '{job_key}' not found",
        )
    return job


@app.delete("/jobs/{job_key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_key: str):
    """Delete a job."""
    deleted = await scheduler.delete_job(job_key)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with key '{job_key}' not found",
        )
    return None
