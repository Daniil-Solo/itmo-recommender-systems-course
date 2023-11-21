from http import HTTPStatus


class RecSysServiceError(Exception):
    """
    Common error for service
    """

    def __init__(self, status_code: int, error_message: str = "") -> None:
        self.error_message = error_message
        self.status_code = status_code
        super().__init__()


class ModelNotFoundError(RecSysServiceError):
    """
    Error accessing a non-existent model
    """

    DEFAULT_MESSAGE = "There is no user with this name"

    def __init__(
        self,
        status_code: int = HTTPStatus.NOT_FOUND,
        error_message: str = DEFAULT_MESSAGE,
    ) -> None:
        super().__init__(status_code, error_message)


class UserNotFoundError(RecSysServiceError):
    """
    Error accessing a non-existent user
    """

    DEFAULT_MESSAGE = "There is no user with this id"

    def __init__(
        self,
        status_code: int = HTTPStatus.NOT_FOUND,
        error_message: str = DEFAULT_MESSAGE,
    ) -> None:
        super().__init__(status_code, error_message)


class NoAuthTokenError(RecSysServiceError):
    """
    Error of missing authorization token
    """

    DEFAULT_MESSAGE = "There is no authorization token"

    def __init__(
        self,
        status_code: int = HTTPStatus.UNAUTHORIZED,
        error_message: str = DEFAULT_MESSAGE,
    ) -> None:
        super().__init__(status_code, error_message)


class IncorrectAuthTokenError(RecSysServiceError):
    """
    Incorrect authorization token error
    """

    DEFAULT_MESSAGE = "Incorrect authorization token"

    def __init__(
        self,
        status_code: int = HTTPStatus.UNAUTHORIZED,
        error_message: str = DEFAULT_MESSAGE,
    ) -> None:
        super().__init__(status_code, error_message)
