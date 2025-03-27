import os
from dotenv import load_dotenv
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('MAESTRO_API_KEY')

# Check if API key is available
if not API_KEY:
    raise ValueError("MAESTRO_API_KEY not found in environment variables")

# Initialize AI21 client with API key from environment
try:
    client = AI21Client(api_key=API_KEY)
    logger.info("AI21 client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI21 client: {str(e)}")
    raise Exception(f"Failed to initialize AI21 client: {str(e)}")