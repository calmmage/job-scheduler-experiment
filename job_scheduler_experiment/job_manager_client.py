#!/usr/bin/env python3
import typer
import requests
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Job Scheduler CLI - Manage and monitor scheduled jobs")
console = Console()
DEFAULT_PORT = 18765


def get_scheduler_url(port: Optional[int] = None) -> str:
    """Get the scheduler URL with optional port override."""
    return f"http://localhost:{port or DEFAULT_PORT}"


@app.command()
def add(
    job_key: str = typer.Argument(..., help="Unique key/identifier for the job"),
    executable_path: Path = typer.Argument(
        ..., help="Path to the Python script to execute"
    ),
    env_path: Path = typer.Argument(..., help="Path to the .env file for the job"),
    interval: Optional[int] = typer.Option(
        None,
        "--interval",
        "-i",
        help="Scheduling interval in seconds (e.g., 300 for 5 minutes). Provide either --interval or --cron.",
    ),
    cron: Optional[str] = typer.Option(
        None,
        "--cron",
        "-c",
        help="Cron schedule string (e.g., '*/5 * * * *' for every 5 minutes). Provide either --interval or --cron.",
    ),
    python_executable: Optional[Path] = typer.Option(
        None,
        "--python-executable",
        "-py",
        help="Optional path to a specific python executable (e.g., from a venv) for this job.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    port: int = typer.Option(DEFAULT_PORT, help="Port of the running scheduler API"),
):
    """Add a new job to the scheduler using either interval or cron.

    Examples:
        job-manager-client add my-job script.py .env --interval 60
        job-manager-client add another-job task.py .env --cron "0 2 * * *"
    """
    scheduler_url = get_scheduler_url(port)
    add_job_url = f"{scheduler_url}/add_job"

    # Validate that exactly one schedule type is provided
    if (interval is None and cron is None) or (
        interval is not None and cron is not None
    ):
        console.print(
            "[red]Error: You must provide exactly one of --interval or --cron.[/red]"
        )
        raise typer.Exit(1)

    if interval is not None and interval <= 0:
        console.print("[red]Error: --interval must be a positive integer.[/red]")
        raise typer.Exit(1)

    # Basic cron validation (server does more thorough check)
    if cron is not None and len(cron.split()) != 5:
        console.print(
            "[red]Warning: Cron string doesn't seem to have 5 parts. Server will perform full validation.[/red]"
        )
        # Let the server handle full cron validation

    try:
        # Resolve paths to absolute paths before sending
        abs_executable_path = str(executable_path.resolve())
        abs_env_path = str(env_path.resolve())
    except Exception as e:
        console.print(f"[red]Error resolving paths: {e}[/red]")
        raise typer.Exit(1)

    payload = {
        "job_key": job_key,
        "path_to_executable": abs_executable_path,
        "path_to_env": abs_env_path,
    }
    # Add the chosen schedule type to the payload
    if interval is not None:
        payload["schedule_interval_seconds"] = interval  # type: ignore[assignment]
    elif cron is not None:
        payload["cron_schedule"] = cron

    # Add optional python executable path if provided
    if python_executable is not None:
        payload["python_executable_path"] = str(python_executable)  # Already resolved

    try:
        response = requests.post(add_job_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        console.print(f"[green]Success: {result['message']}[/green]")
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    port: int = typer.Option(DEFAULT_PORT, help="Port of the running scheduler API"),
):
    """List all jobs in the scheduler."""
    scheduler_url = get_scheduler_url(port)

    try:
        response = requests.get(f"{scheduler_url}/jobs", timeout=10)
        response.raise_for_status()
        jobs = response.json()["jobs"]

        if not jobs:
            console.print("[yellow]No jobs found in the scheduler.[/yellow]")
            return

        table = Table(title="Current Jobs")
        table.add_column("Job Key", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Last Run")
        table.add_column("Next Run")
        table.add_column("Retries", justify="right")

        for job in jobs:
            table.add_row(
                job["job_key"],
                job["status"],
                job["last_run"] or "Never",
                job["next_run"] or "Not scheduled",
                str(job["retry_count"]),
            )

        console.print(table)

    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    job_key: str = typer.Argument(..., help="Job key to check status for"),
    port: int = typer.Option(DEFAULT_PORT, help="Port of the running scheduler API"),
):
    """Get status of a specific job."""
    scheduler_url = get_scheduler_url(port)

    try:
        response = requests.get(f"{scheduler_url}/jobs/{job_key}", timeout=10)
        response.raise_for_status()
        job = response.json()

        error_line = (
            f"Last Error: {job['error_message']}" if job["error_message"] else ""
        )

        console.print(
            Panel.fit(
                f"Status: [magenta]{job['status']}[/magenta]\n"
                f"Last Run: {job['last_run'] or 'Never'}\n"
                f"Next Run: {job['next_run'] or 'Not scheduled'}\n"
                f"Retry Count: {job['retry_count']}\n"
                f"{error_line}",
                title=f"Job Status: {job_key}",
                border_style="blue",
            )
        )

    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def delete(
    job_key: str = typer.Argument(..., help="Job key to delete"),
    port: int = typer.Option(DEFAULT_PORT, help="Port of the running scheduler API"),
):
    """Delete a job from the scheduler."""
    scheduler_url = get_scheduler_url(port)

    try:
        response = requests.delete(f"{scheduler_url}/jobs/{job_key}", timeout=10)
        response.raise_for_status()
        result = response.json()
        console.print(f"[green]Success: {result['message']}[/green]")

    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
