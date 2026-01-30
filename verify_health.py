from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    print(f"Status Code: {response.status_code}")
    print(f"JSON: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

if __name__ == "__main__":
    test_health()
