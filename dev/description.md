# Job Scheduler

A lightweight, reliable job scheduler for macOS that runs Python scripts on a schedule using launchd.

## Overview

The Job Scheduler is designed to be simple yet powerful, providing a robust way to schedule and manage Python scripts on macOS. It leverages launchd for reliable execution and includes features for monitoring, logging, and error handling.

## Key Features

- **Reliable Execution**: Uses launchd for process management and automatic restarts
- **Environment Isolation**: Each job runs in its own virtual environment
- **Centralized Management**: Web API for job management and monitoring
- **Error Handling**: Automatic retries and Telegram notifications
- **Logging**: Comprehensive logging of job execution and errors
- **CLI Interface**: Easy-to-use command-line interface for job management

## Architecture

- **Scheduler**: Core service that manages job execution and scheduling
- **FastAPI Server**: REST API for job management and monitoring
- **MongoDB**: Storage for job configurations and execution history
- **Telegram**: Real-time notifications for job failures
- **launchd**: System service for reliable process management

## Use Cases

- Scheduled data processing
- Automated backups
- Regular system maintenance
- Monitoring and alerting
- Any task that needs to run on a schedule 