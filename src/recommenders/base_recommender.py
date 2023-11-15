from abc import ABC, abstractmethod


class BaseRecommender(ABC):  # pylint: disable=too-few-public-methods
    """
    Базовый класс для рекомендательной системы
    """

    @abstractmethod
    def predict(self, user_id: int) -> list[int]:
        """
        Возвращает список идентификатор объектов, рекомендуемых
        пользователю с идентификатором user_id
        """
        raise NotImplementedError(
            "Данный метод следует реализовать в производном классе"
        )
