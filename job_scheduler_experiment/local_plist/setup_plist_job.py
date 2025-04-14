import os
import plistlib
import subprocess
from pathlib import Path


def setup_plist_job(script_path, env_file=None, hour=12, minute=0, keep_alive=False):
    script_path = Path(script_path).resolve()
    if not script_path.exists() or script_path.suffix != ".py":
        raise ValueError("Script must be a valid .py file")

    job_name = f"com.job_scheduler_experiment.{script_path.stem}"
    venv_python = os.path.join(os.getenv("DEV_ENV_PATH"), "bin/python3")
    if not os.path.exists(venv_python):
        raise ValueError(f"Virtual environment not found at {venv_python}")

    log_dir = Path.home() / "Library/Logs" / "JobSchedulerExperiment"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{script_path.stem}.stdout.log"
    stderr_log = log_dir / f"{script_path.stem}.stderr.log"

    plist = {
        "Label": job_name,
        "ProgramArguments": [venv_python, str(script_path)],
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "RunAtLoad": True,
        "WorkingDirectory": str(script_path.parent),
        "StandardOutPath": str(stdout_log),
        "StandardErrorPath": str(stderr_log),
    }

    if env_file:
        env_file = Path(env_file).resolve()
        if not env_file.exists():
            raise ValueError(f"Environment file not found at {env_file}")
        plist["EnvironmentVariables"] = {"ENV_FILE": str(env_file)}

    if keep_alive:
        plist["KeepAlive"] = True

    plist_path = Path.home() / f"Library/LaunchAgents/{job_name}.plist"
    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)

    subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    print(f"Scheduled {script_path} as {job_name} at {hour:02d}:{minute:02d}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Schedule a Python script with launchd"
    )
    parser.add_argument("script_path", help="Absolute path to the Python script")
    parser.add_argument("--env-file", help="Path to .env file")
    parser.add_argument("--hour", type=int, default=12, help="Hour to run (0-23)")
    parser.add_argument("--minute", type=int, default=0, help="Minute to run (0-59)")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        default=False,
        help="Set KeepAlive=True in the plist to ensure the job restarts if it exits.",
    )
    args = parser.parse_args()
    setup_plist_job(
        args.script_path, args.env_file, args.hour, args.minute, args.keep_alive
    )
