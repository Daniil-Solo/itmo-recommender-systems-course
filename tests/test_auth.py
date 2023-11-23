from fastapi.testclient import TestClient
from src.exceptions import IncorrectAuthTokenError, NoAuthTokenError
from src.config import settings


def test_no_auth_token(client: TestClient) -> None:
    response = client.get("/reco/some_model/1")
    assert response.status_code == 401
    assert response.json() == {'message': NoAuthTokenError.DEFAULT_MESSAGE}


def test_incorrect_format_for_auth_token(client: TestClient) -> None:
    response = client.get("/reco/some_model/1", headers={"Authorization": "Some auth token"})
    assert response.status_code == 401
    assert response.json() == {'message': IncorrectAuthTokenError.DEFAULT_MESSAGE}


def test_incorrect_auth_token(client: TestClient) -> None:
    response = client.get("/reco/some_model/1", headers={"Authorization": "Bearer 111111"})
    assert response.status_code == 401
    assert response.json() == {'message': IncorrectAuthTokenError.DEFAULT_MESSAGE}


def test_correct_auth_token(client: TestClient) -> None:
    response = client.get("/reco/some_model/1", headers={"Authorization": f"Bearer {settings.token}"})
    assert response.status_code != 401
