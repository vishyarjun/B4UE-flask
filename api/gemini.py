import os
from typing import Union, List
from dotenv import load_dotenv
import google.generativeai as genai
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key is available
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize models
    text_model = genai.GenerativeModel('gemini-1.5-flash')
    vision_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    raise Exception(f"Failed to initialize Gemini API: {str(e)}")

def get_text_response(prompt: str) -> str:
    """
    Get a response from the Gemini Pro text model.
    
    Args:
        prompt (str): The text prompt to send to the model
        
    Returns:
        str: The model's response text
    """
    try:
        response = text_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error getting text response from Gemini: {str(e)}")

def get_vision_response(prompt: str, image_data: Union[str, bytes]) -> str:
    """
    Get a response from the Gemini Pro Vision model.
    
    Args:
        prompt (str): The text prompt to send to the model
        image_data (Union[str, bytes]): Either base64 encoded image string or raw bytes
        
    Returns:
        str: The model's response text
    """
    try:
        # Convert bytes to PIL Image if needed
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            # Assume it's base64 encoded
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        
        response = vision_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        raise Exception(f"Error getting vision response from Gemini: {str(e)}")

def analyze_conversation_type(messages: List[dict]) -> str:
    """
    Analyze conversation to determine the type of request.
    
    Args:
        messages (List[dict]): List of conversation messages
        
    Returns:
        str: The determined request type
    """
    try:
        # Join all messages into a single prompt
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        prompt = f"Analyze this conversation and determine the type of request:\n{conversation}\n\nReturn ONLY one of these exact values: defineObjective, defineHealthProfile, collectHealthMetrics, scanFood"
        
        response = get_text_response(prompt)
        return response.strip()
    except Exception as e:
        raise Exception(f"Error analyzing conversation type: {str(e)}")
