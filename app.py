from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil

from rag_chatbot import RAGChatbot

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-chatbot-frontend-phi.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

chatbot = RAGChatbot()

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    paths = []
    for f in files:
        if not f.filename.endswith(".pdf"):
            continue
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        with open(path, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        paths.append(path)

    chatbot.load_documents(paths)
    return {"success": True, "message": f"{len(paths)} document(s) uploaded"}

@app.post("/chat")
async def chat(payload: dict):
    return {"answer": chatbot.chat(payload["message"])}

@app.post("/reset")
async def reset():
    chatbot.reset_conversation()
    return {"success": True}

@app.get("/status")
async def status():
    return {
        "documents": chatbot.db.get_docs_count(),
        "facts": chatbot.db.get_facts_count(),
        "memory": len(chatbot.conversation_history)
    }

@app.get("/health")
@app.head("/health")
def health_check(response: Response):
    return Response(status_code=200)