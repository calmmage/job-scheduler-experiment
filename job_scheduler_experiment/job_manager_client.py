#!/usr/bin/env python3
import argparse
import requests
import sys
from typing import Optional

DEFAULT_PORT = 18765


def get_scheduler_url(port: Optional[int] = None) -> str:
    """Get the scheduler URL with optional port override."""
    return f"http://localhost:{port or DEFAULT_PORT}"


def list_jobs(scheduler_url: str):
    """List all jobs in the scheduler."""
    try:
        response = requests.get(f"{scheduler_url}/jobs", timeout=10)
        response.raise_for_status()
        jobs = response.json()["jobs"]

        if not jobs:
            print("No jobs found in the scheduler.")
            return

        print("\nCurrent Jobs:")
        print("-" * 80)
        for job in jobs:
            print(f"Job Key: {job['job_key']}")
            print(f"Status: {job['status']}")
            print(f"Last Run: {job['last_run'] or 'Never'}")
            print(f"Next Run: {job['next_run'] or 'Not scheduled'}")
            print(f"Retry Count: {job['retry_count']}")
            if job["error_message"]:
                print(f"Last Error: {job['error_message']}")
            print("-" * 80)

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def get_job_status(scheduler_url: str, job_key: str):
    """Get status of a specific job."""
    try:
        response = requests.get(f"{scheduler_url}/jobs/{job_key}", timeout=10)
        response.raise_for_status()
        job = response.json()

        print(f"\nJob Status for '{job_key}':")
        print("-" * 80)
        print(f"Status: {job['status']}")
        print(f"Last Run: {job['last_run'] or 'Never'}")
        print(f"Next Run: {job['next_run'] or 'Not scheduled'}")
        print(f"Retry Count: {job['retry_count']}")
        if job["error_message"]:
            print(f"Last Error: {job['error_message']}")
        print("-" * 80)

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def delete_job(scheduler_url: str, job_key: str):
    """Delete a job from the scheduler."""
    try:
        response = requests.delete(f"{scheduler_url}/jobs/{job_key}", timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"Success: {result['message']}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Manage jobs in the scheduler.")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Common arguments
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port of the running scheduler API (default: {DEFAULT_PORT}).",
    )

    # List jobs command
    list_parser = subparsers.add_parser("list", help="List all jobs")

    # Get job status command
    status_parser = subparsers.add_parser("status", help="Get status of a specific job")
    status_parser.add_argument("job_key", help="Job key to check status for")

    # Delete job command
    delete_parser = subparsers.add_parser("delete", help="Delete a job")
    delete_parser.add_argument("job_key", help="Job key to delete")

    args = parser.parse_args()
    scheduler_url = get_scheduler_url(args.port)

    if args.command == "list":
        list_jobs(scheduler_url)
    elif args.command == "status":
        get_job_status(scheduler_url, args.job_key)
    elif args.command == "delete":
        delete_job(scheduler_url, args.job_key)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
