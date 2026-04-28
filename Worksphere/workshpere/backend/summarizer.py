import os
from dotenv import load_dotenv

load_dotenv()


def summarize_email(email_text, use_gemini=True):
    """Summarize email content"""
    
    if use_gemini:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                prompt = f"""Summarize this email in ONE short sentence (max 15 words). 
Start with "Important:", "Normal:", or "Ignore:".
Format: "Priority: Summary"

Email: {email_text[:800]}"""
                
                response = model.generate_content(prompt)
                result = response.text.strip()
                
                # Ensure it has priority prefix
                if not any(result.startswith(p) for p in ["Important:", "Normal:", "Ignore:"]):
                    result = f"Normal: {result}"
                
                return result
        except Exception as e:
            print(f"Gemini summarization error: {e}")
    
    # Fallback: simple summarization
    try:
        # Take first sentence or first 80 characters
        first_sentence = email_text.split('.')[0][:80]
        if len(first_sentence) < 10:
            first_sentence = email_text[:80]
        return f"Normal: {first_sentence}..."
    except:
        return "Normal: Email received."