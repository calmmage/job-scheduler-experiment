from pydantic_settings import BaseSettings
from pydantic import SecretStr
from aiogram import Bot
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger


class SchedulerConfig(BaseSettings):
    """Scheduler configuration placeholder"""

    port: int = 18765  # Default port for the scheduler API

    mongodb_uri: SecretStr
    mongodb_db_name: str = "job_scheduler"

    # Telegram settings
    telegram_bot_token: SecretStr
    telegram_chat_id: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.bot = Bot(token=config.telegram_bot_token.get_secret_value())
        self.mongo_client = AsyncIOMotorClient(config.mongodb_uri.get_secret_value())
        self.db = self.mongo_client[config.mongodb_db_name]
        logger.info("Scheduler initialized with configuration")

    async def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            # The ismaster command is cheap and does not require auth
            await self.mongo_client.admin.command("ismaster")
            logger.info("MongoDB connection test successful")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
