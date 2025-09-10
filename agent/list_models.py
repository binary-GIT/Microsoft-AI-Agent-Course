import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List all available models
for m in genai.list_models():
    print(m.name, "→ supports:", m.supported_generation_methods)
