from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64
from bs4 import BeautifulSoup
import html
import time
import re
import os

# Cache for emails
CACHE = {
    "data": None,
    "timestamp": 0
}
CACHE_TTL = 30  # seconds

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def get_gmail_service():
    """Get authenticated Gmail service"""
    if not os.path.exists("token.json"):
        raise Exception("Token not found. Run python auth.py first to authenticate with Gmail.")
    
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    return build('gmail', 'v1', credentials=creds)


def extract_headers(msg_data):
    """Extract subject and sender from email headers"""
    headers = msg_data.get('payload', {}).get('headers', [])
    subject = ""
    sender = ""

    for h in headers:
        if h['name'] == 'Subject':
            subject = h['value']
        elif h['name'] == 'From':
            sender = h['value']

    # Clean sender (remove angle brackets and extra text)
    if '<' in sender and '>' in sender:
        match = re.search(r'<(.+?)>', sender)
        if match:
            sender = match.group(1)
    
    return subject[:150] if subject else "No Subject", sender[:100] if sender else "Unknown"


def extract_body(payload):
    """Extract body text from email payload"""
    body_data = ""

    if 'parts' in payload:
        for part in payload['parts']:
            mime = part.get('mimeType', '')
            
            if mime == 'text/plain':
                return part['body'].get('data', '')
            elif mime == 'text/html' and not body_data:
                body_data = part['body'].get('data', '')
    
    # Fallback to main body
    if not body_data:
        body_data = payload.get('body', {}).get('data', '')
    
    return body_data


def clean_email_text(raw_data):
    """Clean and decode email text"""
    try:
        import quopri
        
        # Fix base64 padding
        missing_padding = len(raw_data) % 4
        if missing_padding:
            raw_data += '=' * (4 - missing_padding)
        
        # Decode base64
        decoded_bytes = base64.urlsafe_b64decode(raw_data)
        
        # Handle quoted-printable encoding
        try:
            decoded_bytes = quopri.decodestring(decoded_bytes)
        except:
            pass
        
        # Decode to string
        try:
            decoded = decoded_bytes.decode('utf-8', errors='ignore')
        except:
            decoded = decoded_bytes.decode('latin-1', errors='ignore')
        
        # Remove HTML tags
        soup = BeautifulSoup(decoded, "html.parser")
        
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = soup.get_text(separator=" ")
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove email signatures (common patterns)
        text = re.split(r'(--|Regards,|Thanks,|Best,|Sent from|On.*wrote:)', text)[0]
        
        # Remove excessive special characters
        text = re.sub(r'[^\w\s.,!?₹$@-]', ' ', text)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Remove repetitive words (basic cleanup)
        words = text.split()
        cleaned_words = []
        for w in words:
            if len(cleaned_words) == 0 or w != cleaned_words[-1]:
                cleaned_words.append(w)
        
        text = " ".join(cleaned_words)
        
        # Truncate to reasonable length
        return text[:800]
        
    except Exception as e:
        print(f"Error cleaning email: {e}")
        return ""


def fetch_emails(page_token=None, max_results=20):
    """Fetch emails from Gmail API"""
    # Return cached data if recent
    if CACHE["data"] and time.time() - CACHE["timestamp"] < CACHE_TTL:
        print("Returning cached emails...")
        return CACHE["data"]
    
    print("Fetching fresh emails from Gmail API...")
    
    try:
        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me',
            maxResults=max_results,
            pageToken=page_token
        ).execute()
    except Exception as e:
        print(f"Gmail API error: {e}")
        return {"emails": [], "nextPageToken": None}

    messages = results.get('messages', [])
    next_page_token = results.get('nextPageToken')

    emails = []
    
    for msg in messages:
        try:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            
            subject, sender = extract_headers(msg_data)
            payload = msg_data.get('payload', {})
            
            body_data = extract_body(payload)
            if not body_data:
                continue
            
            cleaned = clean_email_text(body_data)
            
            if len(cleaned) < 20 or "unsubscribe" in cleaned.lower():
                continue
            
            # Basic classification rules
            priority = "Normal"
            category = "General"
            is_spam = False
            
            text_lower = (subject + " " + cleaned).lower()
            
            # Detect important
            important_keywords = ["urgent", "asap", "important", "deadline", "meeting", "review", "approval"]
            if any(word in text_lower for word in important_keywords):
                priority = "Important"
            
            # Detect promotions/spam
            promo_keywords = ["sale", "offer", "discount", "unsubscribe", "win", "free", "newsletter"]
            if any(word in text_lower for word in promo_keywords):
                category = "Promotions"
                is_spam = True
            
            # Detect updates
            elif any(word in text_lower for word in ["order", "account", "login", "security", "alert", "password"]):
                category = "Updates"
            
            # Detect social
            elif any(word in text_lower for word in ["friend", "follow", "like", "comment", "connection"]):
                category = "Social"
            
            emails.append({
                "subject": subject,
                "sender": sender,
                "body": cleaned,
                "priority": priority,
                "category": category,
                "spam": is_spam
            })
            
            if len(emails) >= max_results:
                break
                
        except Exception as e:
            print(f"Error processing email: {e}")
            continue

    result = {
        "emails": emails,
        "nextPageToken": next_page_token
    }
    
    # Save to cache
    CACHE["data"] = result
    CACHE["timestamp"] = time.time()
    
    print(f"✅ Fetched {len(emails)} emails")
    return result


def generate_ai_reply(email_text):
    """Generate AI reply using Gemini"""
    try:
        import google.generativeai as genai
        import os
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "AI reply not available. Please configure GEMINI_API_KEY in .env file."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""Write a short, professional reply to this email. Keep it under 150 words.

Email:
{email_text[:1500]}

Reply:"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"AI reply error: {e}")
        return f"Could not generate reply automatically. Error: {str(e)}"


def summarize_inbox(emails):
    """Generate inbox summary using Gemini"""
    try:
        import google.generativeai as genai
        import os
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Inbox summary not available. Please configure GEMINI_API_KEY in .env file."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare email summaries
        email_summaries = []
        for e in emails[:25]:
            email_summaries.append(f"- {e.get('priority', 'Normal')}: {e.get('subject', 'No subject')[:80]}")
        
        emails_text = "\n".join(email_summaries) if email_summaries else "No emails found"
        
        prompt = f"""Analyze this inbox and provide a concise summary:

{emails_text}

Provide:
1. Total email count
2. Number of important emails
3. Key themes/topics
4. Any urgent matters

Keep it brief and actionable."""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Summarize error: {e}")
        return f"Could not summarize inbox. Error: {str(e)}"