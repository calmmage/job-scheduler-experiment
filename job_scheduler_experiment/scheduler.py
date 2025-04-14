from pathlib import Path
from datetime import datetime
import sys
import time
import traceback
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionError as MongoConnectionError
from aiogram import Bot
from aiogram.exceptions import TelegramError
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("scheduler.log", rotation="1 day")


class JobConfig(BaseModel):
    path_to_executable: Path
    path_to_env: Path
    last_run_timestamp: Optional[datetime] = None
    schedule_period: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # MongoDB settings
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_db_name: str = Field(..., env="MONGODB_DB_NAME")

    # Telegram settings
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(..., env="TELEGRAM_CHAT_ID")


def init_telegram_bot(settings: Settings) -> Bot:
    """Initialize Telegram bot. Crashes if fails."""
    try:
        bot = Bot(token=settings.telegram_bot_token)
        return bot
    except TelegramError as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        raise


def init_mongodb(settings: Settings) -> MongoClient:
    """Initialize MongoDB connection. Retries every 20 minutes if fails."""
    while True:
        try:
            client = MongoClient(settings.mongodb_uri)
            # Test the connection
            client.admin.command("ping")
            return client
        except MongoConnectionError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.info("Retrying in 20 minutes...")
            time.sleep(20 * 60)  # 20 minutes


async def notify_telegram(bot: Bot, chat_id: str, message: str):
    """Send notification to Telegram."""
    try:
        await bot.send_message(chat_id=chat_id, text=message)
    except TelegramError as e:
        logger.error(f"Failed to send Telegram notification: {e}")


def main():
    try:
        # Load settings
        settings = Settings()

        # Initialize connections
        bot = init_telegram_bot(settings)
        mongo_client = init_mongodb(settings)

        db = mongo_client[settings.mongodb_db_name]
        jobs_collection = db.jobs
        logs_collection = db.job_logs

        # TODO: Implement job execution logic

    except Exception as e:
        logger.error(f"Critical error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
