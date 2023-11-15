from typing import Optional

from src.config import settings
from src.exceptions import IncorrectAuthTokenError, NoAuthTokenError


def check_authorization_token(authorization: Optional[str]) -> None:
    """
    Проверяет наличие и валидность токена авторизации
    Если токен не подходит, то выбрасывается ошибка с сообщением
    """
    if not authorization:
        raise NoAuthTokenError()
    try:
        _, token = authorization.split(" ")
    except ValueError as exc:
        raise IncorrectAuthTokenError() from exc
    if token != settings.token:
        raise IncorrectAuthTokenError()
