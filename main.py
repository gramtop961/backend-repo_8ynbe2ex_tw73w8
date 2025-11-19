import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

from database import db, create_document, get_documents

app = FastAPI(title="AI Assistant API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class AskResponse(BaseModel):
    reply: str
    conversation_id: Optional[str] = None

class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    user_id: Optional[str] = None

class ConversationOut(BaseModel):
    id: str
    title: str
    last_message_at: Optional[datetime] = None

class MessageOut(BaseModel):
    role: str
    content: str
    created_at: Optional[datetime] = None


def now_utc():
    return datetime.now(timezone.utc)


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    return d


@app.get("/")
def read_root():
    return {"message": "AI Assistant Backend is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# Simple built-in rules for the assistant (system prompt)
SYSTEM_PROMPT = (
    "You are a helpful, professional AI assistant. "
    "Answer clearly and concisely. Use bullet points when helpful. "
    "If you don't know something, say so.")


def local_answer(user_message: str) -> str:
    """A minimal rule-based fallback assistant if no external LLM is configured."""
    text = user_message.strip()
    if not text:
        return "Could you please share more details?"

    lower = text.lower()
    if any(k in lower for k in ["hello", "hi", "hey"]):
        return "Hello! How can I help you today?"
    if lower.startswith("who are you") or "what are you" in lower:
        return "I'm your AI assistant here to help with explanations, brainstorming, and guidance."
    if lower.startswith("help") or "how do i" in lower:
        return "Happy to help. Tell me your goal and what you've tried so far."

    # Default concise echo-style response
    return (
        "Here are some ways I can help:\n"
        "- Explain concepts step by step\n"
        "- Draft emails, posts, and summaries\n"
        "- Outline plans and checklists\n"
        "- Analyze ideas and suggest improvements\n\n"
        f"Your request: {text}")


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    # Try to route to external LLM if configured via environment
    provider = os.getenv("LLM_PROVIDER", "local").lower()

    reply: str

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
        try:
            import requests
            # Use Chat Completions API compatible format (o3-mini or gpt-4o-mini if available)
            url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.message}
                ],
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
            }
            import json
            r = requests.post(url, json=payload, timeout=60, headers=headers)
            r.raise_for_status()
            data = r.json()
            reply = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Fall back to local responder on any error
            reply = local_answer(req.message) + f"\n\n(Note: OpenAI request failed: {str(e)[:100]})"
    else:
        # Local fallback
        reply = local_answer(req.message)

    conv_id = req.conversation_id or os.getenv("DEFAULT_CONVERSATION_ID", "default")

    # Persist conversation and message if database is available
    try:
        if db is not None:
            # Save user message
            create_document("message", {
                "conversation_id": conv_id,
                "role": "user",
                "content": req.message,
                "created_at": now_utc()
            })
            # Save assistant reply
            create_document("message", {
                "conversation_id": conv_id,
                "role": "assistant",
                "content": reply,
                "created_at": now_utc()
            })
            # Upsert conversation metadata (title inferred from first user message if not set)
            title = (req.message[:40] + "…") if len(req.message) > 40 else req.message
            db["conversation"].update_one(
                {"_id": conv_id},
                {"$setOnInsert": {"_id": conv_id, "title": title or "New conversation", "created_at": now_utc()},
                 "$set": {"last_message_at": now_utc()}},
                upsert=True
            )
    except Exception:
        # Non-fatal if DB not configured
        pass

    return AskResponse(reply=reply, conversation_id=conv_id)


@app.post("/api/conversations", response_model=ConversationOut)
async def create_conversation(payload: CreateConversationRequest):
    if db is None:
        # If db not available, create ephemeral id
        cid = str(uuid4())
        return ConversationOut(id=cid, title=payload.title or "New conversation", last_message_at=None)

    cid = str(uuid4())
    doc = {
        "_id": cid,
        "title": payload.title or "New conversation",
        "user_id": payload.user_id,
        "created_at": now_utc(),
        "last_message_at": None,
    }
    db["conversation"].insert_one(doc)
    return ConversationOut(id=cid, title=doc["title"], last_message_at=None)


@app.get("/api/conversations", response_model=List[ConversationOut])
async def list_conversations() -> List[ConversationOut]:
    if db is None:
        return []
    items = list(db["conversation"].find({}, {"title": 1, "last_message_at": 1}).sort("last_message_at", -1))
    result: List[ConversationOut] = []
    for it in items:
        it = serialize_doc(it)
        result.append(ConversationOut(id=it["id"], title=it.get("title", "Conversation"), last_message_at=it.get("last_message_at")))
    return result


@app.get("/api/conversations/{conversation_id}/messages", response_model=List[MessageOut])
async def get_messages(conversation_id: str) -> List[MessageOut]:
    if db is None:
        return []
    msgs = list(db["message"].find({"conversation_id": conversation_id}).sort("created_at", 1))
    out: List[MessageOut] = []
    for m in msgs:
        out.append(MessageOut(role=m.get("role", "user"), content=m.get("content", ""), created_at=m.get("created_at")))
    return out


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
