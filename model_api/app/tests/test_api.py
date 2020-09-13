from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_health_endpoint_returns_200():
    # When
    response = client.get("/health")
    # Then
    assert response.status_code == 200
    
def test_prediction_endpoint_returns_float():
    # When
    test_image = "../../model_package/cnn_model/cnn_model/data/test/img_10.jpg"
    response = client.post(
        "/predict", files={"file": ("test_image", open(test_image, "rb"))}
        )
    # Then
    assert type(response.json()["prediction"]) == float
    
def test_prediction_endpoint_returns_correct_prediction():
    # When
    test_image = "../../model_package/cnn_model/cnn_model/data/test/img_4.jpg"
    response = client.post(
        "/predict", files={"file": ("test_image", open(test_image, "rb"))}
        )
    # Then
    assert response.json()["prediction"] == float(6)
    
def test_version_endpoint_returns_correct_model_version():
    # When
    response = client.get("/version")
    # Then
    assert response.json()["model_version"] == "1.0.0"
    
def test_version_endpoint_returns_correct_api_version():
    # When
    response = client.get("/version")
    # Then
    assert response.json()["api_version"] == "1.0.0"