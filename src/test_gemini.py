import os
from pathlib import Path
from dotenv import load_dotenv


# Find the .env file in the main project folder
env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("Gemini API key loaded successfully!")
else:
    print("Gemini API key NOT found.")