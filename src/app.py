from src.scheduler import Scheduler, SchedulerConfig
from fastapi import FastAPI
import os
from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables
load_dotenv()

app = FastAPI()
config = SchedulerConfig(
    mongodb_uri=SecretStr(os.getenv("MONGODB_URI", "")),
    telegram_bot_token=SecretStr(os.getenv("TELEGRAM_BOT_TOKEN", "")),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
)
scheduler = Scheduler(config)


@app.get("/")
async def root():
    return {"message": "Hello World"}
