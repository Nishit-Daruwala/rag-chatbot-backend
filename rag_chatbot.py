# rag_chatbot.py - OPENAI EMBEDDINGS VERSION (LIGHTWEIGHT)

import os
import uuid
import time
from typing import List, Tuple

import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ================= CONFIG =================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

DOC_SIM_THRESHOLD = 0.32
FACT_SIM_THRESHOLD = 0.45

MAX_MEMORY_TURNS = 5
QDRANT_BATCH_SIZE = 50

# OpenAI Embedding Model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# ================= PDF HANDLING =================

def load_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    text = text.replace("\x00", "")
    return text.encode("utf-8", "ignore").decode("utf-8")


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ================= QDRANT DATABASE =================

class QdrantVectorDB:
    def __init__(self, embedding_dim: int):
        print("☁️ Connecting to Qdrant Cloud...")

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )

        # distinct collection names for OpenAI vectors
        self.docs = "documents_openai"
        self.facts = "facts_openai"

        self._create_collection(self.docs, embedding_dim)
        self._create_collection(self.facts, embedding_dim)

        print("✓ Connected to Qdrant Cloud")
        print("Documents:", self.get_docs_count())
        print("Facts:", self.get_facts_count())

    def _create_collection(self, name: str, dim: int):
        collections = [c.name for c in self.client.get_collections().collections]
        if name not in collections:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        for i in range(0, len(texts), QDRANT_BATCH_SIZE):
            batch_texts = texts[i:i + QDRANT_BATCH_SIZE]
            batch_embs = embeddings[i:i + QDRANT_BATCH_SIZE]

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={"text": text}
                )
                for text, emb in zip(batch_texts, batch_embs)
            ]

            self.client.upsert(collection_name=self.docs, points=points)
            time.sleep(0.2)

    def add_fact(self, text: str, emb: List[float]):
        self.client.upsert(
            collection_name=self.facts,
            points=[PointStruct(id=str(uuid.uuid4()), vector=emb, payload={"text": text})]
        )

    def search_docs(self, emb: List[float], k=8):
        return [(h.payload["text"], h.score)
                for h in self.client.query_points(self.docs, query=emb, limit=k).points]

    def search_facts(self, emb: List[float], k=5):
        return [(h.payload["text"], h.score)
                for h in self.client.query_points(self.facts, query=emb, limit=k).points]

    def get_docs_count(self):
        return self.client.count(self.docs).count

    def get_facts_count(self):
        return self.client.count(self.facts).count


# ================= RAG CHATBOT =================

class RAGChatbot:
    def __init__(self):
        print("🤖 Initializing RAG Chatbot (OpenAI Embeddings)...")
        # No local model loading here! Very fast and low memory.
        
        self.db = QdrantVectorDB(EMBEDDING_DIM)
        self.conversation_history = []

        print("✓ Chatbot ready!")

    def get_embedding(self, text: str) -> List[float]:
        # Use OpenAI API for embedding
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        # Use OpenAI API for batch embeddings
        # OpenAI supports batching, but for huge lists we might still want to chunk calls
        # client.embeddings.create handles lists gracefully up to a limit (usually 2048 dims or batch size)
        
        all_embeddings = []
        BATCH_LIMIT = 50 # Safe batch size for API calls
        
        for i in range(0, len(texts), BATCH_LIMIT):
            batch = texts[i:i+BATCH_LIMIT]
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            all_embeddings.extend([d.embedding for d in response.data])
            
        return all_embeddings

    def load_documents(self, pdf_paths: List[str]):
        chunks = []
        for path in pdf_paths:
            text = load_pdf(path)
            chunks.extend(chunk_text(text))

        if not chunks:
            return

        print(f"Generating embeddings for {len(chunks)} chunks via OpenAI...")
        embeddings = self.get_embeddings_batch(chunks)

        self.db.add_documents(chunks, embeddings)

    def contextualize(self, query: str) -> str:
        if not self.conversation_history:
            return query
        last = self.conversation_history[-1]["user"]
        if any(p in query.lower() for p in ["it", "this", "that", "they"]):
            return f"{last}. Follow-up: {query}"
        return query

    def chat(self, query: str) -> str:
        # 1. Contextualize the query based on history
        query = self.contextualize(query)

        # 2. Get embedding from API
        emb = self.get_embedding(query)

        facts = [t for t, s in self.db.search_facts(emb) if s >= FACT_SIM_THRESHOLD]
        docs = [t for t, s in self.db.search_docs(emb) if s >= DOC_SIM_THRESHOLD]

        context = "\n".join(facts + docs)

        if context.strip():
            print("💡 Using RAG Context")
            prompt = f"You are a helpful assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer strictly based on the context provided. If the context does not contain the answer, do NOT makeup information, just state that you cannot answer from the documents."
        else:
            print("🧠 Using General Knowledge")
            prompt = f"You are a helpful assistant.\n\nQuestion:\n{query}\n\nAnswer the question based on your general knowledge."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()

        self.conversation_history.append({
            "user": query,
            "assistant": answer
        })

        self.conversation_history = self.conversation_history[-MAX_MEMORY_TURNS:]

        return answer

    def reset_conversation(self):
        self.conversation_history = []
