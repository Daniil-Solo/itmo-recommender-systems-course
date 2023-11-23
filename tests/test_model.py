from fastapi.testclient import TestClient
from src.exceptions import ModelNotFoundError
from src.config import settings
from src.recommenders.random_recommender import RandomRecommender


def test_incorrect_model_name(client: TestClient) -> None:
    user_id = 1
    response = client.get(f"/reco/some_model_name/{user_id}", headers={"Authorization": f"Bearer {settings.token}"})
    assert response.status_code == 404
    assert response.json() == {'message': ModelNotFoundError.DEFAULT_MESSAGE}


def test_correct_model_name(client: TestClient) -> None:
    user_id = 1
    response = client.get(f"/reco/{RandomRecommender.MODEL_NAME}/{user_id}", headers={"Authorization": f"Bearer {settings.token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user_id
    assert len(data["items"]) == settings.n_returned_items
    assert all(isinstance(item_id, int) for item_id in data["items"])
