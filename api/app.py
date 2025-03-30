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
        # Start timing the request
        start_time = time.time()
        
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

        # Limit the number of ingredients to analyze (to prevent timeouts)
        max_ingredients = 10  
        if len(ingredients) > max_ingredients:
            ingredients = ingredients[:max_ingredients]
            logger.warning(f"Limiting analysis to first {max_ingredients} ingredients")

        # Create a simplified prompt for ingredient analysis
        system_prompt = """You are a nutrition expert. Analyze ingredients for health impact based on the provided health data.
For each ingredient, determine if it's good or bad considering:
- Dietary requirements (e.g., vegetarian, vegan)
- Allergies
- Health conditions (e.g., cholesterol, diabetes, fatty liver)

Return ONLY a JSON array with this format:
[
  {
    "name": "ingredient name",
    "classification": "good" or "bad",
    "key_impact": "brief one-sentence impact"
  }
]

Be extremely concise. No explanations outside the JSON."""

        # Create a simplified analysis prompt that extracts key health information
        health_summary = {}
        
        # Extract only essential health data to reduce payload size
        if "dietaryRequirement" in health_data:
            health_summary["diet"] = health_data["dietaryRequirement"]
            
        # Extract allergies
        if "allergies" in health_data and health_data["allergies"]:
            health_summary["allergies"] = health_data["allergies"]
            
        # Extract health conditions - limit to most critical ones
        if "healthConditions" in health_data and health_data["healthConditions"]:
            # Only include the most nutrition-relevant conditions
            relevant_conditions = []
            for condition in health_data["healthConditions"]:
                condition_lower = condition.lower()
                if any(keyword in condition_lower for keyword in ["diabetes", "cholesterol", "heart", "liver", "kidney", "pressure", "celiac"]):
                    relevant_conditions.append(condition)
            
            if relevant_conditions:
                health_summary["conditions"] = relevant_conditions
            
        # Extract only critical lab values - limit to just a few key ones
        if "additionalHealthData" in health_data and health_data["additionalHealthData"]:
            abnormal_labs = {}
            critical_tests = ["glucose", "cholesterol", "triglycerides", "a1c"]
            
            for test, data in health_data["additionalHealthData"].items():
                if any(critical in test.lower() for critical in critical_tests):
                    if "result" in data and "referenceInterval" in data:
                        try:
                            result = float(data["result"])
                            ref_interval = data["referenceInterval"]
                            
                            # Simplified reference interval check
                            if "-" in ref_interval:
                                low, high = map(float, ref_interval.split("-"))
                                if result < low or result > high:
                                    abnormal_labs[test] = {
                                        "result": result,
                                        "units": data.get("units", "")
                                    }
                        except (ValueError, TypeError):
                            pass
            
            if abnormal_labs:
                health_summary["abnormal_labs"] = abnormal_labs
        
        # Create the analysis prompt with the simplified health summary
        analysis_prompt = f"""Health profile: {json.dumps(health_summary)}
Ingredients to analyze: {json.dumps([i.get("name", "") for i in ingredients])}
Analyze each ingredient's impact on this specific health profile. Return ONLY the JSON array."""

        # Check if we've already spent too much time preparing the request
        elapsed_time = time.time() - start_time
        if elapsed_time > 10:  # If preparation took more than 10 seconds, use fallback
            logger.warning(f"Request preparation took {elapsed_time} seconds, using fallback response")
            return generate_fallback_response(ingredients)

        # Call AI21 API with reduced tokens and explicit timeout
        messages = [
            SystemMessage(content=system_prompt, role="system"),
            UserMessage(content=analysis_prompt, role="user")
        ]
        
        try:
            # Set a timeout for the API call
            import threading
            
            response_container = {"response": None, "error": None, "completed": False}
            
            def api_call():
                try:
                    response = client.chat.completions.create(
                        messages=messages,
                        model="jamba-large",
                        temperature=0.1,
                        max_tokens=800  
                    )
                    response_container["response"] = response
                    response_container["completed"] = True
                except Exception as e:
                    response_container["error"] = str(e)
                    response_container["completed"] = True
            
            # Start API call in a thread
            thread = threading.Thread(target=api_call)
            thread.daemon = True  # Make thread a daemon so it doesn't block process exit
            thread.start()
            
            # Wait for a maximum of 20 seconds (reduced from 25)
            max_wait_time = 20
            wait_interval = 0.5
            total_waited = 0
            
            while not response_container["completed"] and total_waited < max_wait_time:
                time.sleep(wait_interval)
                total_waited += wait_interval
                
                # Check if we're getting close to the timeout
                if total_waited >= max_wait_time * 0.8:
                    logger.warning(f"API call taking too long ({total_waited} seconds), preparing fallback")
            
            if not response_container["completed"]:
                # API call is still running after timeout
                logger.warning(f"AI21 API call timed out after {total_waited} seconds")
                return generate_fallback_response(ingredients)
            
            if response_container["error"]:
                logger.error(f"AI21 API error: {response_container['error']}")
                return generate_fallback_response(ingredients)
            
            response = response_container["response"]
            
            # Get the response text
            analysis = response.choices[0].message.content
            
            # Check if we're approaching the function timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > 25:  # If we've already spent 25+ seconds, use fallback
                logger.warning(f"Processing taking too long ({elapsed_time} seconds), using fallback")
                return generate_fallback_response(ingredients)
            
        except Exception as e:
            logger.error(f"AI21 API error: {str(e)}")
            return generate_fallback_response(ingredients)

        # Try to parse the response as JSON
        try:
            # Clean up the response to ensure it's valid JSON
            cleaned_response = analysis.strip()
            
            # Remove any markdown formatting - simplified cleanup
            if "```" in cleaned_response:
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "")
                
            cleaned_response = cleaned_response.strip()
            
            # Check if we're approaching the function timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > 27:  # If we've already spent 27+ seconds, use fallback
                logger.warning(f"JSON parsing taking too long ({elapsed_time} seconds), using fallback")
                return generate_fallback_response(ingredients)
            
            # Parse and validate the response
            analyzed_ingredients = json.loads(cleaned_response)
            
            # Ensure we have a list
            if not isinstance(analyzed_ingredients, list):
                if isinstance(analyzed_ingredients, dict) and "ingredients" in analyzed_ingredients:
                    analyzed_ingredients = analyzed_ingredients.get("ingredients", [])
                else:
                    logger.warning("Invalid response format, using fallback")
                    return generate_fallback_response(ingredients)
            
            # Format the response
            formatted_response = {
                "ingredients": [],
                "summary": {
                    "safe_to_consume": True,
                    "overall_impact": "Analysis completed successfully"
                }
            }
            
            # Count bad ingredients
            bad_count = 0
            
            # Process each ingredient
            for item in analyzed_ingredients:
                ingredient = {
                    "name": item.get("name", "").strip(),
                    "classification": item.get("classification", "unknown"),
                    "impacts": [
                        {
                            "metric": "health",
                            "effect": item.get("key_impact", ""),
                            "severity": "negative" if item.get("classification") == "bad" else "positive"
                        }
                    ],
                    "warnings": [] if item.get("classification") != "bad" else ["May have negative health effects"]
                }
                
                if ingredient["classification"] == "bad":
                    bad_count += 1
                
                formatted_response["ingredients"].append(ingredient)
            
            # Update safety assessment
            if bad_count > len(formatted_response["ingredients"]) / 2:
                formatted_response["summary"]["safe_to_consume"] = False
                formatted_response["summary"]["overall_impact"] = "High proportion of harmful ingredients"
            
            return jsonify(formatted_response)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Raw response: {analysis}")
            return generate_fallback_response(ingredients)
            
    except Exception as e:
        logger.error(f"Error in ingredient analysis endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

def generate_fallback_response(ingredients):
    """
    Generate a fallback response when AI analysis fails or times out
    """
    fallback_response = {
        "ingredients": [],
        "summary": {
            "safe_to_consume": True,
            "overall_impact": "Analysis could not be completed. Please try again with fewer ingredients."
        }
    }
    
    # Add basic analysis for each ingredient
    for item in ingredients:
        ingredient_name = item.get("name", "").strip()
        fallback_response["ingredients"].append({
            "name": ingredient_name,
            "classification": "unknown",
            "impacts": [
                {
                    "metric": "health",
                    "effect": "Could not analyze this ingredient",
                    "severity": "neutral"
                }
            ],
            "warnings": []
        })
    
    return jsonify(fallback_response)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)
