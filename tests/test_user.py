from fastapi.testclient import TestClient
from src.exceptions import UserNotFoundError
from src.config import settings


def test_incorrect_user_id(client: TestClient) -> None:
    user_id = 10 ** 10
    response = client.get(f"/reco/random/{user_id}", headers={"Authorization": f"Bearer {settings.token}"})
    assert response.status_code == 404
    assert response.json() == {'message': UserNotFoundError.DEFAULT_MESSAGE}


def test_correct_user_id(client: TestClient) -> None:
    user_id = 1
    response = client.get(f"/reco/random/{user_id}", headers={"Authorization": f"Bearer {settings.token}"})
    assert response.status_code == 200
