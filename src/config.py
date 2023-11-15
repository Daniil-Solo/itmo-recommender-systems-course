from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Настройки для сервиса"""

    token: str = Field(alias="TOKEN", default="TEST_TOKEN")
    n_returned_items: int = Field(alias="N_RETURNED_ITEMS", default=10)


settings = Settings()
