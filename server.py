import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import google.generativeai as genai
import PyPDF2
import docx
import textract
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config
print("hello")
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

class OpenAIService:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_post(self, prompt, platform='linkedin', max_tokens=300):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful social media content assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        
        except RateLimitError:
            logger.error("OpenAI API rate limit exceeded")
            raise
        except APIConnectionError:
            logger.error("Failed to connect to OpenAI API")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class GeminiService:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_post(self, prompt, platform='linkedin', max_tokens=300):
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

PLATFORM_PROMPTS = {
    'linkedin': "Write a professional, engaging LinkedIn post using the AIDA method. ",
    'instagram': "Create a concise, compelling Instagram post with appropriate hashtags. ",
    'facebook': "Craft a friendly and informative Facebook post that encourages engagement. ",
    'twitter': "Write a punchy, concise tweet that captures attention in 280 characters. "
}

def extract_text_from_file(file):
    """Extract text from various file types"""
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.pdf':
            reader = PyPDF2.PdfReader(file)
            text = ' '.join([page.extract_text() for page in reader.pages])
        elif file_ext in ['.docx', '.doc']:
            text = textract.process(file).decode('utf-8')
        elif file_ext == '.txt':
            text = file.read().decode('utf-8')
        else:
            raise ValueError("Unsupported file type")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

@app.route('/generate-post', methods=['POST'])
def generate_post():
    if 'idea' not in request.form:
        return jsonify({"error": "No post idea provided"}), 400
    
    idea = request.form['idea']
    platform = request.form.get('platform', 'linkedin')
    ai_model = request.form.get('ai_model', 'openai')
    style_text = ""

    if 'styleFile' in request.files:
        style_file = request.files['styleFile']
        if style_file and Config.is_valid_file(style_file.filename):
            style_text = extract_text_from_file(style_file)

    try:
        base_prompt = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS['linkedin'])
        style_instruction = (
            "Please mimic the following writing style and tone: " + 
            (style_text[:1000] if style_text else "")
        )
        
        full_prompt = f"{base_prompt}\n{style_instruction}\nIdea: {idea}"
        
        if ai_model == 'gemini':
            ai_service = GeminiService(Config.GEMINI_API_KEY)
        else:
            ai_service = OpenAIService(Config.OPENAI_API_KEY)
        
        generated_post = ai_service.generate_post(full_prompt, platform)
        
        return jsonify({"post": generated_post})
    
    except Exception as e:
        logger.error(f"Error generating post: {e}")
        return jsonify({
            "error": "Could not generate post", 
            "details": str(e)
        }), 500

@app.route('/')
def home():
    return "LIFTU Backend is running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)