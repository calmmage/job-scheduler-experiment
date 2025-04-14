from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


class Job(BaseModel):
    job_key: str = Field(..., description="Unique identifier for the job")
    path_to_executable: str = Field(
        ..., description="Path to the script to be executed"
    )
    path_to_env: Optional[str] = Field(
        None, description="Path to environment file with variables"
    )
    schedule_interval_seconds: Optional[int] = Field(
        None, description="Interval in seconds between job runs"
    )
    cron_schedule: Optional[str] = Field(
        None, description="Cron expression for scheduling"
    )
    python_executable_path: Optional[str] = Field(
        None, description="Path to Python executable"
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING, description="Current status of the job"
    )
    last_run: Optional[datetime] = Field(
        None, description="Timestamp of last execution"
    )
    next_run: Optional[datetime] = Field(
        None, description="Scheduled timestamp for next execution"
    )
    retry_count: int = Field(
        default=0, description="Number of times the job has been retried"
    )
    error_message: Optional[str] = Field(
        None, description="Last error message if job failed"
    )

    class Config:
        schema_extra = {
            "example": {
                "job_key": "backup_job",
                "path_to_executable": "/scripts/backup.py",
                "path_to_env": "/scripts/.env",
                "schedule_interval_seconds": 3600,
                "cron_schedule": None,
                "python_executable_path": "/usr/bin/python3",
                "status": "pending",
                "last_run": None,
                "next_run": None,
                "retry_count": 0,
                "error_message": None,
            }
        }


class JobCreate(BaseModel):
    job_key: str
    path_to_executable: str
    path_to_env: Optional[str] = None
    schedule_interval_seconds: Optional[int] = None
    cron_schedule: Optional[str] = None
    python_executable_path: Optional[str] = None


class JobUpdate(BaseModel):
    path_to_executable: Optional[str] = None
    path_to_env: Optional[str] = None
    schedule_interval_seconds: Optional[int] = None
    cron_schedule: Optional[str] = None
    python_executable_path: Optional[str] = None
    status: Optional[JobStatus] = None
