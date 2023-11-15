from fastapi import Path

from src.exceptions import UserNotFoundError


def get_user_id(user_id: int = Path()) -> int:
    """
    Извлекает из пути идентификатор пользователя
    Проверяет на корректность значения
    """
    if user_id > 10**9:
        raise UserNotFoundError()
    return user_id
