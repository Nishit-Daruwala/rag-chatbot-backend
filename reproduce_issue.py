import os
import sys
from unittest.mock import MagicMock

# Mock VectorDB before importing RAGChatbot to avoid locking/initialization issues
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()

# Now import
from rag_chatbot import RAGChatbot

def test_contextualization():
    print("Initializing Chatbot (with mocked DB)...")
    bot = RAGChatbot()
    bot.vector_db = MagicMock() # Ensure it's mocked on the instance too
    
    # Setup conversation history similar to the user's screenshot
    bot.conversation_history = [
        {
            "user": "what is perceptron ?",
            "assistant": "A Perceptron is simply composed of a single layer of TLUs, with each TLU connected to all the inputs. It is a type of artificial neuron."
        },
        {
            "user": "what is data object?",
            "assistant": "A data object is a collection of attributes that describe an entity."
        }
    ]
    
    query = "is there any types of it ?"
    print(f"\nTesting Query: '{query}'")
    print(f"History Depth: {len(bot.conversation_history)}")
    
    rewritten = bot.contextualize_query(query)
    
    print(f"\nOriginal:  {query}")
    print(f"Rewritten: {rewritten}")
    
    if "data object" in rewritten.lower():
        print("✅ SUCCESS: Query was contextualized correctly.")
    else:
        print("❌ FAILURE: Query was NOT contextualized correctly.")

if __name__ == "__main__":
    test_contextualization()
