import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # New Gemini API key
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB file upload limit
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

    @staticmethod
    def is_valid_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS