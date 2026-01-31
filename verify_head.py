from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_head_health():
    print("Testing HEAD /health...")
    response = client.head("/health")
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {response.headers}")
    # Note: HEAD response body is ignored by client usually, but status should be 200
    if response.status_code == 200:
        print("SUCCESS: HEAD request allowed.")
    else:
        print(f"FAILURE: HEAD request returned {response.status_code}")

if __name__ == "__main__":
    test_head_health()
