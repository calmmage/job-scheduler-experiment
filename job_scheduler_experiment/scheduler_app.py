from typing import Optional

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

from .scheduler import Settings


class SchedulerApp:
    def __init__(self):
        self._settings: Optional[Settings] = None
        self._bot: Optional[Bot] = None
        self._mongo_client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._fastapi_app: Optional[FastAPI] = None

    @property
    def settings(self) -> Settings:
        if not self._settings:
            raise RuntimeError("Settings not initialized")
        return self._settings

    @property
    def bot(self) -> Bot:
        if not self._bot:
            raise RuntimeError("Bot not initialized")
        return self._bot

    @property
    def mongo_client(self) -> MongoClient:
        if not self._mongo_client:
            raise RuntimeError("MongoDB client not initialized")
        return self._mongo_client

    @property
    def db(self) -> Database:
        if not self._db:
            raise RuntimeError("Database not initialized")
        return self._db

    @property
    def scheduler(self) -> AsyncIOScheduler:
        if not self._scheduler:
            raise RuntimeError("Scheduler not initialized")
        return self._scheduler

    @property
    def fastapi_app(self) -> FastAPI:
        if not self._fastapi_app:
            raise RuntimeError("FastAPI app not initialized")
        return self._fastapi_app

    async def initialize(self) -> None:
        """Initialize all components of the application."""
        try:
            # Initialize settings
            self._settings = Settings()  # type: ignore

            # Initialize bot
            self._bot = self._init_telegram_bot()

            # Initialize MongoDB
            self._mongo_client = self._init_mongodb()
            self._db = self._mongo_client[self._settings.mongodb_db_name]

            # Initialize scheduler
            self._scheduler = AsyncIOScheduler()

            # Initialize FastAPI app
            self._fastapi_app = FastAPI()

            logger.info("SchedulerApp initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SchedulerApp: {e}")
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Shutdown all components of the application."""
        if self._scheduler and self._scheduler.running:
            logger.info("Stopping scheduler...")
            self._scheduler.shutdown()

        if self._bot and self._bot.session:
            logger.info("Closing bot session...")
            await self._bot.session.close()

        if self._mongo_client:
            logger.info("Closing MongoDB connection...")
            self._mongo_client.close()

        logger.info("SchedulerApp shutdown complete")

    def _init_telegram_bot(self) -> Bot:
        """Initialize the Telegram bot."""
        return Bot(token=self.settings.telegram_bot_token)

    def _init_mongodb(self) -> MongoClient:
        """Initialize MongoDB connection."""
        return MongoClient(
            self.settings.mongodb_uri,
            server_api=self.settings.mongodb_server_api,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000,
        )
