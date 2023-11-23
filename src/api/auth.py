from typing import Optional

from src.config import settings
from src.exceptions import IncorrectAuthTokenError, NoAuthTokenError


def check_authorization_token(authorization: Optional[str]) -> None:
    """
    Checks the availability and validity of the authorization token
    If the token is not correct, then an error message is raised
    """
    if not authorization:
        raise NoAuthTokenError()
    try:
        _, token = authorization.split(" ")
    except ValueError as exc:
        raise IncorrectAuthTokenError() from exc
    if token != settings.token:
        raise IncorrectAuthTokenError()
