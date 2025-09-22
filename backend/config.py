"""
Configuration settings for Deep Researcher Agent
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Deep Researcher Agent"""
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # HuggingFace Configuration (alternative to OpenAI)
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    # Vector Store Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", "10000000"))  # 10MB
    
    # Document Processing
    SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "txt,pdf,docx").split(",")
    MAX_DOCUMENTS_PER_QUERY = int(os.getenv("MAX_DOCUMENTS_PER_QUERY", "50"))
    
    # Report Generation
    REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "./data/reports")
    DEFAULT_REPORT_FORMAT = os.getenv("DEFAULT_REPORT_FORMAT", "pdf")
    
    # CORS Configuration
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    
    # Query Processing
    MAX_SUB_QUERIES = 5
    MIN_RELEVANCE_SCORE = 0.3
    TOP_K_DOCUMENTS = 10
    
    # Synthesis Configuration
    MAX_ANSWER_LENGTH = 15000  # Increased for comprehensive research answers
    INCLUDE_CITATIONS = True
    SYNTHESIZER_BACKEND = os.getenv("SYNTHESIZER_BACKEND", "local").lower()  # local | openai | huggingface
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        if not cls.OPENAI_API_KEY and not cls.HUGGINGFACE_API_KEY:
            print("Warning: No API key provided for OpenAI or HuggingFace")
            return False
        
        # Create necessary directories
        os.makedirs(cls.VECTOR_STORE_PATH, exist_ok=True)
        os.makedirs(cls.REPORT_OUTPUT_DIR, exist_ok=True)
        os.makedirs("./data/documents", exist_ok=True)
        
        return True

# Global config instance
config = Config()
