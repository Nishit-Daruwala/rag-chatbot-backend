
import sys
from unittest.mock import MagicMock, patch

# Mock modules that might require dependencies or keys
sys.modules["openai"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

# Patch QdrantClient before importing
with patch("qdrant_client.QdrantClient") as MockClient:
    # Setup the mock instance
    mock_instance = MockClient.return_value
    # Setup query_points return value to act like QueryResponse with points list
    mock_point = MagicMock()
    mock_point.payload = {"text": "test document"}
    mock_point.score = 0.9
    
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    
    mock_instance.query_points.return_value = mock_response
    mock_instance.get_collections.return_value.collections = []

    # Now import the module under test
    # We need to make sure we modify sys.path if necessary or just run from backend dir
    import rag_chatbot
    
    # Initialize DB (will use MockClient)
    db = rag_chatbot.QdrantVectorDB(384)
    
    # Test search_docs
    print("Testing search_docs...")
    results = db.search_docs([0.1]*384)
    
    # Verification
    print(f"Results: {results}")
    
    if mock_instance.query_points.called:
        print("✅ SUCCESS: query_points was called.")
    else:
        print("❌ FAILURE: query_points was NOT called.")
        
    if mock_instance.search.called:
        print("❌ FAILURE: search was called (should be query_points).")
    else:
        print("✅ SUCCESS: search was NOT called.")

