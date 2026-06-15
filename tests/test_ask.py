def test_ask_returns_dataset_answer(client):
    response = client.post("/ask", json={"question": "How do I reverse a list in Python?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert len(data["answer"]) > 0


def test_ask_rejects_short_question(client):
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422
