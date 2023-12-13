from enum import Enum
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class ModeEnum(str, Enum):
    """
    Mode of application
    """

    TEST = "test"
    PRODUCTION = "prod"


class Settings(BaseSettings):
    """Settings for the service"""

    mode: ModeEnum = Field(alias="MODE", default=ModeEnum.TEST)
    token: str = Field(alias="TOKEN", default="TEST_TOKEN")
    n_returned_items: int = Field(alias="N_RETURNED_ITEMS", default=10)


load_dotenv()
settings = Settings()
