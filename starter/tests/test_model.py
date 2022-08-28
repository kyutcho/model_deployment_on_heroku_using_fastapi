import sys

from fastapi.testclient import TestClient

sys.path.append('../../../starter')

from main import app

client = TestClient(app)

def test_get_path():
    r = client.get("/")