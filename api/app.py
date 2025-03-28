from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
from PIL import Image
import io
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from .gemini import get_text_response, get_vision_response
from functools import wraps
# Import AI21 client library
from ai21 import AI21Client
from ai21.models.chat.chat_message import SystemMessage, UserMessage, AssistantMessage

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for agent types
AGENT_DEFINE_OBJECTIVE = "defineObjective"
AGENT_DEFINE_HEALTH_PROFILE = "defineHealthProfile"
AGENT_COLLECT_HEALTH_METRICS = "collectHealthMetrics"
AGENT_SCAN_FOOD = "scanFood"

# Get API token from environment variable
API_TOKEN = os.getenv('API_TOKEN')
if not API_TOKEN:
    logger.warning("API_TOKEN not set in environment variables. Using default token for development.")
    API_TOKEN = "your-default-token-here"  # Only for development

def require_api_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No Authorization header"}), 401
        
        try:
            token_type, token = auth_header.split()
            if token_type.lower() != 'bearer':
                return jsonify({"error": "Invalid token type"}), 401
            if token != API_TOKEN:
                return jsonify({"error": "Invalid token"}), 401
        except ValueError:
            return jsonify({"error": "Invalid Authorization header format"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

# Function to get AI21 client
def get_ai21_client():
    api_key = os.getenv("AI21_API_KEY")
    
    if not api_key:
        logger.error("AI21_API_KEY not found in environment variables")
        return None
        
    # Create the client using the API key
    return AI21Client(api_key=api_key)

@app.route('/api/image-scan', methods=['POST'])
@require_api_token
def analyze_image():
    """
    Endpoint for analyzing images using Gemini Vision
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file
        image_data = image_file.read()
        
        # Construct a detailed prompt for health report format
        base_prompt = """Analyze this image and provide a JSON response. If it's a health report, follow this exact format for each test:
{
  "TEST_NAME": {
    "referenceInterval": "MIN - MAX",
    "result": NUMERIC_VALUE,
    "units": "UNIT" or null
  }
}

For other types of images:
- Food images: Include nutritional information
- Ingredient lists: Provide a structured list

Format the response as valid JSON without any additional text or markdown."""

        prompt = request.form.get('prompt', base_prompt)

        # Analyze image with Gemini Vision
        analysis = get_vision_response(prompt, image_data)
        
        # Try to parse the response as JSON
        try:
            import json
            # Clean up the response to ensure it's valid JSON
            # Remove any markdown formatting if present
            cleaned_response = analysis.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Parse the cleaned response
            parsed_response = json.loads(cleaned_response)
            
            # Additional validation for health report format
            if any(isinstance(v, dict) and 'referenceInterval' in v for v in parsed_response.values()):
                # This appears to be a health report, validate the format
                formatted_response = {}
                for key, value in parsed_response.items():
                    if isinstance(value, dict):
                        formatted_response[key] = {
                            "referenceInterval": value.get("referenceInterval", ""),
                            "result": float(value.get("result", 0)) if value.get("result") is not None else None,
                            "units": value.get("units")
                        }
                    else:
                        formatted_response[key] = {
                            "referenceInterval": "",
                            "result": float(value) if value is not None else None,
                            "units": None
                        }
                return jsonify(formatted_response)
            
            return jsonify(parsed_response)
        except json.JSONDecodeError:
            # If parsing fails, wrap the response in a simple structure
            return jsonify({
                'analysis': analysis.strip()
            })

    except Exception as e:
        logger.error(f"Error in image analysis endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/food-scan', methods=['POST'])
@require_api_token
def scan_food_image():
    """
    Endpoint for analyzing food images or ingredient lists using Gemini Vision.
    Returns a list of ingredients for both cases.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file
        image_data = image_file.read()
        
        # Construct a detailed prompt for ingredient analysis
        base_prompt = """Analyze this image and determine if it's a food photo or ingredient list.
If it's neither, respond with type "other". Provide a JSON response in this exact format:
{
  "type": "ingredient_list" or "food_photo" or "other",
  "ingredients": [
    {
      "name": "ingredient name",
      "amount": "quantity if specified or null",
      "unit": "unit of measurement if specified or null"
    }
  ],
  "confidence": "high" or "medium" or "low"
}

If the image is a list of ingredients:
- Set type as "ingredient_list"
- Extract each ingredient with its amount and unit if specified
- Set confidence as "high"

If the image is a photo of food:
- Set type as "food_photo"
- Make your best guess of the main ingredients
- Set confidence based on how certain you are of the ingredients
- Include common ingredients that would typically be used

If the image is neither food nor ingredients:
- Set type as "other"
- Return an empty ingredients list
- Set confidence as "high"

Format the response as valid JSON without any additional text or markdown."""

        prompt = request.form.get('prompt', base_prompt)

        # Analyze image with Gemini Vision
        analysis = get_vision_response(prompt, image_data)
        
        # Try to parse the response as JSON
        try:
            import json
            # Clean up the response to ensure it's valid JSON
            cleaned_response = analysis.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Parse the cleaned response
            parsed_response = json.loads(cleaned_response)
            
            # Validate and format the response
            formatted_response = {
                "type": parsed_response.get("type", "other"),
                "ingredients": [],
                "confidence": parsed_response.get("confidence", "low")
            }
            
            # Only process ingredients if type is food_photo or ingredient_list
            if formatted_response["type"] in ["food_photo", "ingredient_list"]:
                # Format ingredients
                for ingredient in parsed_response.get("ingredients", []):
                    if isinstance(ingredient, dict):
                        formatted_ingredient = {
                            "name": ingredient.get("name", "").strip(),
                            "amount": ingredient.get("amount"),
                            "unit": ingredient.get("unit")
                        }
                    else:
                        # Handle case where ingredient is just a string
                        formatted_ingredient = {
                            "name": str(ingredient).strip(),
                            "amount": None,
                            "unit": None
                        }
                    if formatted_ingredient["name"]:  # Only add if name is not empty
                        formatted_response["ingredients"].append(formatted_ingredient)
            
            return jsonify(formatted_response)
            
        except json.JSONDecodeError:
            # If parsing fails, return as "other" type with empty ingredients
            return jsonify({
                "type": "other",
                "ingredients": [],
                "confidence": "low"
            })

    except Exception as e:
        logger.error(f"Error in food scan endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/analyze-ingredients', methods=['POST'])
@require_api_token
def analyze_ingredients():
    """
    Analyze ingredients list against health data using AI21 Maestro
    Returns impact analysis and classification for each ingredient
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        health_data = data.get('health_data')
        ingredients = data.get('ingredients')

        if not health_data or not ingredients:
            return jsonify({'error': 'Both health_data and ingredients are required'}), 400

        # Get AI21 client
        client = get_ai21_client()
        if not client:
            return jsonify({'error': 'AI21 client initialization failed'}), 500

        # Split ingredients into smaller chunks if too many
        chunk_size = 20
        ingredient_chunks = [ingredients[i:i + chunk_size] for i in range(0, len(ingredients), chunk_size)]
        
        all_analyzed_ingredients = []
        
        for chunk in ingredient_chunks:
            # Create a detailed prompt for ingredient analysis
            system_prompt = """You are a health and nutrition expert. Analyze each ingredient's impact on health metrics.
For each ingredient, determine:
1. Overall classification (good/bad)
2. Key impacts on health
3. Any warnings

Provide analysis in this JSON format:
{
  "ingredients": [
    {
      "name": "ingredient",
      "classification": "good" or "bad",
      "impacts": [{"metric": "health metric", "effect": "brief effect", "severity": "positive/negative"}],
      "warnings": ["key warnings"] or []
    }
  ]
}

IMPORTANT: Be concise. Format as valid JSON only."""

            # Create the analysis prompt for this chunk
            analysis_prompt = f"""Health Data:
{json.dumps(health_data, indent=2)}

Ingredients to analyze:
{json.dumps(chunk, indent=2)}

Analyze each ingredient's impact on these health metrics. Consider interactions and cumulative effects.
Format the response as specified JSON without any markdown formatting or additional text."""

            # Call AI21 API with increased max tokens
            messages = [
                SystemMessage(content=system_prompt, role="system"),
                UserMessage(content=analysis_prompt, role="user")
            ]
            
            response = client.chat.completions.create(
                messages=messages,
                model="jamba-large",
                temperature=0.1,
                max_tokens=2000
            )

            # Get the response text
            analysis = response.choices[0].message.content

            # Try to parse the response as JSON
            try:
                # Clean up the response to ensure it's valid JSON
                cleaned_response = analysis.strip()
                
                # Remove any markdown formatting
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response.split('\n', 1)[1]  # Remove first line
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response.rsplit('\n', 1)[0]  # Remove last line
                if cleaned_response.startswith('json'):
                    cleaned_response = cleaned_response.split('\n', 1)[1]  # Remove json tag
                    
                cleaned_response = cleaned_response.strip()
                
                # Parse and validate the response
                parsed_response = json.loads(cleaned_response)
                
                # Add ingredients from this chunk to the overall list
                chunk_ingredients = parsed_response.get("ingredients", [])
                all_analyzed_ingredients.extend(chunk_ingredients)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in chunk: {str(e)}")
                logger.error(f"Raw chunk response: {analysis}")
                continue  # Skip this chunk if parsing fails
        
        if not all_analyzed_ingredients:
            return jsonify({
                "error": "Failed to analyze any ingredients",
                "details": "All chunks failed to parse"
            }), 500
            
        # Create the final response with all analyzed ingredients
        formatted_response = {
            "ingredients": [],
            "summary": {
                "safe_to_consume": True,  # Will be updated based on analysis
                "overall_impact": "Analysis completed successfully"
            }
        }
        
        # Format each ingredient analysis
        bad_ingredient_count = 0
        very_bad_ingredient_count = 0
        
        for ingredient in all_analyzed_ingredients:
            formatted_ingredient = {
                "name": ingredient.get("name", "").strip(),
                "classification": ingredient.get("classification", "unknown"),
                "impacts": [],
                "warnings": ingredient.get("warnings", []),
                "recommendations": ingredient.get("recommendations", [])
            }
            
            # Count bad and very bad ingredients
            if formatted_ingredient["classification"] == "bad":
                bad_ingredient_count += 1
            elif formatted_ingredient["classification"] == "very bad":
                very_bad_ingredient_count += 1
            
            # Format impacts
            for impact in ingredient.get("impacts", []):
                if isinstance(impact, dict):
                    formatted_impact = {
                        "metric": impact.get("metric", "").strip(),
                        "effect": impact.get("effect", "").strip(),
                        "severity": impact.get("severity", "neutral")
                    }
                    if formatted_impact["metric"] and formatted_impact["effect"]:
                        formatted_ingredient["impacts"].append(formatted_impact)
            
            formatted_response["ingredients"].append(formatted_ingredient)
        
        # Update safety assessment based on bad ingredient counts
        if very_bad_ingredient_count > 0:
            formatted_response["summary"]["safe_to_consume"] = False
            formatted_response["summary"]["overall_impact"] = "Contains very harmful ingredients, consumption not recommended"
        elif bad_ingredient_count > len(formatted_response["ingredients"]) / 3:  # If more than 1/3 are bad
            formatted_response["summary"]["safe_to_consume"] = False
            formatted_response["summary"]["overall_impact"] = "High proportion of harmful ingredients, limited consumption recommended"
        else:
            formatted_response["summary"]["overall_impact"] = "Moderate to low health impact, consume in moderation"
        
        return jsonify(formatted_response)
            
    except Exception as e:
        logger.error(f"Error in ingredient analysis endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)
