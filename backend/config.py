import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Ollama Configuration
    OLLAMA_URL = os.environ.get('OLLAMA_URL') or 'http://127.0.0.1:11434'
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL') or 'llama3.1:8b'
    
    # API Configuration
    API_PORT = int(os.environ.get('API_PORT') or 5500)
