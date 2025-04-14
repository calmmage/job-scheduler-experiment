import argparse
import requests
import sys
from pathlib import Path
import json


def add_job(
    scheduler_url: str, job_key: str, executable_path: str, env_path: str, interval: int
):
    """Sends a request to the scheduler API to add a new job."""

    # Resolve paths to absolute paths before sending
    try:
        abs_executable_path = str(Path(executable_path).resolve())
        abs_env_path = str(Path(env_path).resolve())
    except Exception as e:
        print(f"Error resolving paths: {e}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "job_key": job_key,
        "path_to_executable": abs_executable_path,
        "path_to_env": abs_env_path,
        "schedule_interval_seconds": interval,
    }

    add_job_url = f"{scheduler_url.rstrip('/')}/add_job"

    print(f"Sending request to {add_job_url} with payload:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(
            add_job_url, json=payload, timeout=10
        )  # 10 second timeout

        print(f"\nResponse Status Code: {response.status_code}")

        try:
            response_json = response.json()
            print("Response Body:")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("Response Body (non-JSON):")
            print(response.text)

        if not response.ok:
            print(
                f"\nError: Received status code {response.status_code}", file=sys.stderr
            )
            sys.exit(1)

    except requests.exceptions.ConnectionError as e:
        print(
            f"\nError: Could not connect to the scheduler at {add_job_url}",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(
            f"\nError: Request timed out connecting to {add_job_url}", file=sys.stderr
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(
            f"\nError: An unexpected error occurred during the request: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a job to the scheduler via its API."
    )

    parser.add_argument("job_key", help="Unique key/identifier for the job.")
    parser.add_argument(
        "executable_path", help="Absolute path to the Python script to execute."
    )
    parser.add_argument("env_path", help="Absolute path to the .env file for the job.")
    parser.add_argument("interval", type=int, help="Scheduling interval in seconds.")
    parser.add_argument(
        "--scheduler-url",
        default="http://localhost:8000",
        help="URL of the running scheduler API (default: http://localhost:8000).",
    )

    args = parser.parse_args()

    add_job(
        scheduler_url=args.scheduler_url,
        job_key=args.job_key,
        executable_path=args.executable_path,
        env_path=args.env_path,
        interval=args.interval,
    )

    print("\nJob add request sent successfully.")
