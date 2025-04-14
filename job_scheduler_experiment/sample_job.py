from dotenv import load_dotenv

import os
from datetime import datetime
import requests
from loguru import logger


def main():
    env_file = os.getenv("ENV_FILE", ".env")
    if not load_dotenv(env_file):
        logger.error(f"Failed to load .env file: {env_file}")
        return
    # Try importing optional dependencies
    calmlib_import_success = False
    try:
        import calmlib

        calmlib_import_success = True
    except ImportError:
        pass

    dev_env_import_success = False
    try:
        import dev_env

        dev_env_import_success = True
    except ImportError:
        pass

    telegram_downloader_import_success = False
    try:
        import telegram_downloader

        telegram_downloader_import_success = True
    except ImportError:
        pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    method = os.getenv("METHOD")
    web_app_script_id = os.getenv("WEB_APP_SCRIPT_ID")

    if not method or not web_app_script_id:
        logger.error(
            "Missing required environment variables: METHOD and/or WEB_APP_SCRIPT_ID"
        )
        return

    target_url = f"https://script.google.com/macros/s/{web_app_script_id}/exec"

    payload = {
        "method": method,
        "timestamp": timestamp,
        "calmlib_import_success": calmlib_import_success,
        "dev_env_import_success": dev_env_import_success,
        "telegram_downloader_import_success": telegram_downloader_import_success,
    }

    try:
        response = requests.post(target_url, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully sent request to {target_url}")
        logger.info(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send request: {e}")


if __name__ == "__main__":
    main()
