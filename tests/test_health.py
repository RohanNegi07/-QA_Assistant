def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape(client):
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert "pipeline_ready" in data
    assert "version" in data
