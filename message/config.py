import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.parent.absolute()

DATA_DIR = Path(BASE_DIR, "data")
NOTEBOOKS_DIR = Path(BASE_DIR, "notebooks")
QUERIES_DIR = Path(BASE_DIR, "queries")
TESTS_DIR = Path(BASE_DIR, "tests")
PROMPTS_DIR = Path(BASE_DIR, "prompts")

# create dirs
BASE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
QUERIES_DIR.mkdir(parents=True, exist_ok=True)
TESTS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    class Config:
        env_file = f"{BASE_DIR}/.env"


@lru_cache()
def get_settings() -> Settings:
    logging.info("Loading config settings from the environment...")
    return Settings()
