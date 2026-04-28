from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# Import from local modules
from gmail_service import fetch_emails, generate_ai_reply, summarize_inbox
from summarizer import summarize_email

# Load environment variables
load_dotenv()

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini AI configured")
    else:
        gemini_model = None
        print("⚠️ GEMINI_API_KEY not found in .env")
except ImportError:
    gemini_model = None
    print("⚠️ google-generativeai not installed")

app = FastAPI(title="WorkSphere Assistant API", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DATA MODELS ====================
class Document(BaseModel):
    id: str
    type: str
    title: str
    owner: str
    content: str
    createdAt: str

class Reminder(BaseModel):
    id: str
    text: str
    time: str
    createdAt: str
    completed: bool = False

class ChatRequest(BaseModel):
    message: str
    mode: str = "qa"
    context: Optional[Dict[str, Any]] = None

# ==================== DATA STORAGE ====================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DOCS_FILE = DATA_DIR / "companyDocs.json"
REMINDERS_FILE = DATA_DIR / "reminders.json"

company_docs: List[Document] = []
reminders: List[Reminder] = []

def load_data():
    global company_docs, reminders
    
    if DOCS_FILE.exists():
        try:
            with open(DOCS_FILE, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
                company_docs = [Document(**d) for d in docs_data]
            print(f"✅ Loaded {len(company_docs)} documents")
        except Exception as e:
            print(f"Error loading docs: {e}")
            company_docs = []
    
    if REMINDERS_FILE.exists():
        try:
            with open(REMINDERS_FILE, 'r', encoding='utf-8') as f:
                rem_data = json.load(f)
                reminders = [Reminder(**r) for r in rem_data]
            print(f"✅ Loaded {len(reminders)} reminders")
        except Exception as e:
            print(f"Error loading reminders: {e}")
            reminders = []
    
    # Initialize with default documents if empty
    if not company_docs:
        company_docs = [
            Document(
                id="1", 
                type='policy', 
                title='Leave Policy 2025', 
                owner='HR',
                content='18 days annual leave, 6 casual days. WFH up to 3 days/week.',
                createdAt=datetime.now().isoformat()
            ),
            Document(
                id="2", 
                type='announce', 
                title='Q2 Hiring Plan', 
                owner='Talent',
                content='Hiring 15 engineers, 5 PMs starting April 15. Referral bonus ₹50k.',
                createdAt=datetime.now().isoformat()
            ),
            Document(
                id="3", 
                type='announce', 
                title='Wellness Program', 
                owner='HR',
                content='₹10,000 wellness allowance, 4 mental health days per year.',
                createdAt=datetime.now().isoformat()
            ),
        ]
        save_docs()

def save_docs():
    with open(DOCS_FILE, 'w', encoding='utf-8') as f:
        json.dump([d.dict() for d in company_docs], f, indent=2, ensure_ascii=False)

def save_reminders():
    with open(REMINDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump([r.dict() for r in reminders], f, indent=2, ensure_ascii=False)

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "WorkSphere Assistant API",
        "version": "2.0.0",
        "status": "running",
        "gemini_configured": gemini_model is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gemini_configured": gemini_model is not None,
        "documents_count": len(company_docs),
        "reminders_count": len(reminders)
    }

# ==================== GMAIL ENDPOINTS ====================

@app.get("/emails")
async def get_emails(page_token: Optional[str] = Query(None), max_results: int = Query(20)):
    """Fetch and summarize emails from Gmail"""
    try:
        data = fetch_emails(page_token=page_token, max_results=max_results)
        emails = data.get("emails", [])
        next_token = data.get("nextPageToken")

        results = []
        for email in emails:
            # Get summary using summarizer
            summary_text = summarize_email(email.get("body", ""))
            
            # Parse priority from summary
            priority = "Normal"
            clean_summary = summary_text
            
            if summary_text and ":" in summary_text:
                parts = summary_text.split(":", 1)
                possible_priority = parts[0].strip().capitalize()
                if possible_priority in ["Important", "Normal", "Ignore"]:
                    priority = possible_priority
                    clean_summary = parts[1].strip() if len(parts) > 1 else summary_text
            
            results.append({
                "id": str(uuid.uuid4())[:8],
                "sender": email.get("sender", "Unknown"),
                "subject": email.get("subject", "No Subject"),
                "summary": clean_summary[:200] if clean_summary else "No summary available",
                "priority": priority,
                "category": email.get("category", "General"),
                "body": email.get("body", "")[:500],
                "spam": email.get("spam", False)
            })
        
        return {
            "success": True,
            "emails": results,
            "nextPageToken": next_token,
            "count": len(results)
        }
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "emails": []}
        )

@app.get("/summary")
async def inbox_summary():
    """Get AI-powered inbox summary"""
    try:
        data = fetch_emails(max_results=30)
        emails = data.get("emails", [])
        
        if not emails:
            return {"success": True, "summary": "No emails found in your inbox."}
        
        summary = summarize_inbox(emails)
        return {"success": True, "summary": summary}
    except Exception as e:
        print(f"Summary error: {e}")
        return {"success": False, "summary": f"Error generating summary: {str(e)}"}

@app.post("/reply")
async def generate_reply(request: Request):
    """Generate AI reply for an email"""
    try:
        data = await request.json()
        email_text = data.get("text", "")
        subject = data.get("subject", "")
        sender = data.get("sender", "")
        
        full_text = f"Subject: {subject}\nFrom: {sender}\n\n{email_text}"
        reply = generate_ai_reply(full_text)
        
        return {"success": True, "reply": reply}
    except Exception as e:
        print(f"Reply generation error: {e}")
        return {"success": False, "reply": f"Error: {str(e)}"}

# ==================== KNOWLEDGE BASE ENDPOINTS ====================

@app.get("/api/documents")
async def get_documents(search: Optional[str] = Query(None)):
    docs = company_docs.copy()
    if search:
        search_lower = search.lower()
        docs = [d for d in docs if search_lower in d.title.lower() or search_lower in d.content.lower()]
    return {"success": True, "documents": [d.dict() for d in docs]}

@app.post("/api/documents")
async def create_document(request: Request):
    try:
        data = await request.json()
        new_doc = Document(
            id=str(uuid.uuid4()),
            type=data.get('type', 'policy'),
            title=data['title'],
            owner=data.get('owner', 'Team'),
            content=data['content'],
            createdAt=datetime.now().isoformat()
        )
        company_docs.append(new_doc)
        save_docs()
        return {"success": True, "document": new_doc.dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    global company_docs
    company_docs = [d for d in company_docs if d.id != doc_id]
    save_docs()
    return {"success": True}

# ==================== REMINDERS ENDPOINTS ====================

@app.get("/api/reminders")
async def get_reminders():
    return {"success": True, "reminders": [r.dict() for r in reversed(reminders)]}

@app.post("/api/reminders")
async def create_reminder(request: Request):
    try:
        data = await request.json()
        new_reminder = Reminder(
            id=str(uuid.uuid4()),
            text=data['text'],
            time=data.get('time', 'today'),
            createdAt=datetime.now().isoformat(),
            completed=False
        )
        reminders.append(new_reminder)
        save_reminders()
        return {"success": True, "reminder": new_reminder.dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str):
    global reminders
    reminders = [r for r in reminders if r.id != reminder_id]
    save_reminders()
    return {"success": True}

# ==================== CHAT ENDPOINT ====================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """General chat with context awareness"""
    message = request.message
    
    # Check for reminder intent
    remind_match = re.search(r'remind\s+me\s+to\s+(.+?)(?:\s+at\s+|\s+on\s+|\s+tomorrow\s+)(.+?)$', message.lower())
    if remind_match:
        task, when = remind_match.groups()
        new_reminder = Reminder(
            id=str(uuid.uuid4()),
            text=task.strip(),
            time=when.strip(),
            createdAt=datetime.now().isoformat(),
            completed=False
        )
        reminders.append(new_reminder)
        save_reminders()
        return {
            "success": True,
            "response": f"✅ Reminder set! I'll remind you to **{task.strip()}** {when.strip()}.",
            "reminderAdded": new_reminder.dict()
        }
    
    # Simple response if Gemini not configured
    if not gemini_model:
        return {
            "success": True,
            "response": "🤖 I'm here to help! To enable AI features, please add your GEMINI_API_KEY to the .env file. Meanwhile, you can still use Gmail integration and reminders.",
            "sources": []
        }
    
    try:
        # Search knowledge base for relevant docs
        relevant_docs = []
        for doc in company_docs:
            if (doc.title.lower() in message.lower() or 
                any(word in message.lower() for word in doc.content.lower().split()[:3])):
                relevant_docs.append(doc)
        
        context = ""
        if relevant_docs:
            context = "\n\n**Relevant knowledge base entries:**\n"
            for doc in relevant_docs[:2]:
                context += f"• {doc.title}: {doc.content[:200]}\n"
        
        prompt = f"""You are WorkSphere Assistant, an AI workplace assistant.
{context}

User: {message}

Provide a helpful, concise response. Use emojis where appropriate."""
        
        response = await gemini_model.generate_content_async(prompt)
        
        return {
            "success": True,
            "response": response.text,
            "sources": [d.dict() for d in relevant_docs[:3]]
        }
    except Exception as e:
        print(f"Chat error: {e}")
        return {
            "success": True,
            "response": f"I encountered an error: {str(e)}. Please try again.",
            "sources": []
        }

@app.get("/test")
async def test():
    """Test endpoint"""
    return {"message": "API is working!"}

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup():
    load_data()
    print("\n" + "="*50)
    print("🚀 WorkSphere Assistant Backend Started")
    print("="*50)
    print(f"📍 Server: http://localhost:3001")
    print(f"🤖 Gemini AI: {'✓' if gemini_model else '✗ (Add GEMINI_API_KEY to .env)'}")
    print(f"📚 Documents: {len(company_docs)}")
    print(f"⏰ Reminders: {len(reminders)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001, reload=True)