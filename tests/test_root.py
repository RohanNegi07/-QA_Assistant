def test_root_returns_welcome_message(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Python Q&A Assistant API is running",
        "docs": "/docs",
    }
