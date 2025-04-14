from pathlib import Path
from datetime import datetime
import sys
import time
import traceback
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
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
    except TelegramAPIError as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        raise


def init_mongodb(settings: Settings) -> MongoClient:
    """Initialize MongoDB connection. Retries every 20 minutes if fails."""
    while True:
        try:
            # Create a new client and connect to the server
            client = MongoClient(
                settings.mongodb_uri,
                server_api=ServerApi("1"),
                serverSelectionTimeoutMS=5000,  # 5 seconds timeout
                connectTimeoutMS=10000,  # 10 seconds connection timeout
                socketTimeoutMS=45000,  # 45 seconds socket timeout
            )

            # Test the connection
            # The ismaster command is cheap and does not require auth.
            client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
            return client

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.info("Retrying in 20 minutes...")
            time.sleep(20 * 60)  # 20 minutes


async def notify_telegram(bot: Bot, chat_id: str, message: str):
    """Send notification to Telegram."""
    try:
        await bot.send_message(chat_id=chat_id, text=message)
    except TelegramAPIError as e:
        logger.error(f"Failed to send Telegram notification: {e}")


def main():
    try:
        # Load settings
        settings = Settings()

        # Initialize connections
        bot = init_telegram_bot(settings)
        mongo_client = init_mongodb(settings)

        db = mongo_client[settings.mongodb_db_name]

        # Initialize collections with indexes
        jobs_collection = db.jobs
        logs_collection = db.job_logs

        # Create indexes
        jobs_collection.create_index("last_run_timestamp")
        jobs_collection.create_index("schedule_period")

        logs_collection.create_index("timestamp")
        logs_collection.create_index("job_key")
        logs_collection.create_index("status")

        logger.info("MongoDB collections and indexes initialized")

        # TODO: Implement job execution logic

    except Exception as e:
        logger.error(f"Critical error: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
