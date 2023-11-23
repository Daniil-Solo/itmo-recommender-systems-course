from fastapi import Path

from src.exceptions import UserNotFoundError


def get_user_id(user_id: int = Path()) -> int:
    """
    Extracts user ID from the path
    Checks for the correctness of the value
    """
    if user_id > 10**9:
        raise UserNotFoundError()
    return user_id
