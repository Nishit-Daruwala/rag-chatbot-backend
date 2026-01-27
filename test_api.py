import os
import io
import pytest
from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

# ================= TESTS =================

def test_status_endpoint():
    """Test if the status endpoint returns 200 and correct structure."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "documentsLoaded" in data
    assert "conversationTurns" in data

def test_chat_empty_message():
    """Test chat with empty message should fail gracefully (400 or handled)."""
    response = client.post("/chat", json={"message": ""})
    # app.py explicitly raises 400 for empty message
    assert response.status_code == 400
    assert response.json()["detail"] == "Message is required"

def test_chat_missing_field():
    """Test chat with missing payload field."""
    response = client.post("/chat", json={})
    assert response.status_code == 400

@patch("rag_chatbot.RAGChatbot.chat")
def test_chat_success(mock_chat):
    """Test successful chat flow."""
    mock_chat.return_value = "Hello, I am a bot."
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Hello, I am a bot."

def test_upload_no_file():
    """Test upload without files."""
    response = client.post("/upload", files={})
    assert response.status_code == 422 # FastAPI default for missing required field

def test_upload_invalid_file_type():
    """Test uploading a non-PDF file."""
    file_content = b"This is a text file."
    files = {"files": ("test.txt", file_content, "text/plain")}
    response = client.post("/upload", files=files)
    # app.py logic: checks is_pdf, if not pdf continues. 
    # If no valid pdfs found, raises 400.
    assert response.status_code == 400
    assert response.json()["detail"] == "No valid PDF files found"

@patch("rag_chatbot.RAGChatbot.load_documents")
def test_upload_valid_pdf(mock_load):
    """Test uploading a valid PDF file."""
    # Create a dummy PDF signature
    pdf_content = b"%PDF-1.4\n..."
    files = {"files": ("test.pdf", pdf_content, "application/pdf")}
    
    with patch("builtins.open"), patch("shutil.copyfileobj"): 
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        assert response.json()["success"] is True

@patch("rag_chatbot.client.chat.completions.create")
def test_openai_api_failure_handling(mock_create):
    """Test how the chatbot handles OpenAI API failures (Simulated Crash)."""
    # Simulate an exception from OpenAI
    mock_create.side_effect = Exception("OpenAI API Unavailable")
    
    # We expect the app to handle this gracefully and return a polite message
    response = client.post("/chat", json={"message": "trigger error"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "trouble connecting" in data["answer"]

