# agent/llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import HuggingFaceChat
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Primary: Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Fallback: HuggingFace
hf_llm = HuggingFaceChat(
    repo_id="google/flan-t5-small",  # choose any available HuggingFace chat model
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    model_kwargs={"temperature": 0.3, "max_new_tokens": 200}
)

def safe_invoke(prompt: str) -> str:
    try:
        response = gemini_llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        elif hasattr(response, "text"):
            return response.text
        else:
            return str(response)
    except Exception as e:
        print(f"[WARN] Gemini failed ({e}), falling back to HuggingFace...")
        try:
            return hf_llm.invoke(prompt)
        except Exception as e2:
            return f"[ERROR] Both Gemini and HuggingFace failed: {e2}"
