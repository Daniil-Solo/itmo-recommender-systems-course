from http import HTTPStatus


class RecSysServiceError(Exception):
    """
    Общая ошибка для приложения
    """

    def __init__(self, status_code: int, error_message: str = "") -> None:
        self.error_message = error_message
        self.status_code = status_code
        super().__init__()


class ModelNotFoundError(RecSysServiceError):
    """
    Ошибка обращения к несуществующей модели
    """

    def __init__(
        self,
        status_code: int = HTTPStatus.NOT_FOUND,
        error_message: str = "Модели с таким названием не существует",
    ) -> None:
        super().__init__(status_code, error_message)


class UserNotFoundError(RecSysServiceError):
    """
    Ошибка обращения к несуществующему пользователю
    """

    def __init__(
        self,
        status_code: int = HTTPStatus.NOT_FOUND,
        error_message: str = "Пользователя с таким id не существует",
    ) -> None:
        super().__init__(status_code, error_message)


class NoAuthTokenError(RecSysServiceError):
    """
    Ошибка отсутствия аутентификационного токена
    """

    def __init__(
        self,
        status_code: int = HTTPStatus.UNAUTHORIZED,
        error_message: str = "Отсутствует токен авторизации",
    ) -> None:
        super().__init__(status_code, error_message)


class IncorrectAuthTokenError(RecSysServiceError):
    """
    Ошибка неправильного аутентификационного токена
    """

    def __init__(
        self,
        status_code: int = HTTPStatus.UNAUTHORIZED,
        error_message: str = "Неверный токен авторизации",
    ) -> None:
        super().__init__(status_code, error_message)
