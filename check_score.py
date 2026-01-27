
import sys
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def check_scores():
    print("Initializing...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    query = "what is nominal ?"
    print(f"\nQuery: '{query}'")
    
    emb = embedder.encode([query])[0].tolist()
    
    print("\n--- Document Matches ---")
    results = client.query_points("documents", query=emb, limit=5).points
    
    if not results:
        print("No matches found.")
    
    for hit in results:
        print(f"Score: {hit.score:.4f} | Text preview: {hit.payload['text'][:100]}...")

if __name__ == "__main__":
    check_scores()
