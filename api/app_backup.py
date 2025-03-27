from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from maestro import client as ai21_client
from gemini import get_text_response, get_vision_response
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for text-based interactions using AI21
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message']
        context = data.get('context', [])  # Optional previous messages

        # Process with AI21
        response = ai21_client.chat.create(
            model="j2-ultra",
            messages=[*context, {"role": "user", "content": message}]
        )

        return jsonify({
            'response': response.choices[0].message.content,
            'context': [*context, 
                       {"role": "user", "content": message},
                       {"role": "assistant", "content": response.choices[0].message.content}]
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/image-scan', methods=['POST'])
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
        prompt = request.form.get('prompt', "This image has either list of ingredients or a photo of food or a lab report or user health report. Analyze and provide response as lists.")

        # Analyze image with Gemini Vision
        analysis = get_vision_response(prompt, image_data)

        return jsonify({
            'analysis': analysis
        })

    except Exception as e:
        logger.error(f"Error in image analysis endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """
    Endpoint for analyzing text using Gemini
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        prompt = data.get('prompt', "Analyze this text and provide insights.")

        # Analyze text with Gemini
        analysis = get_text_response(f"{prompt}\n\nText: {text}")

        return jsonify({
            'analysis': analysis
        })

    except Exception as e:
        logger.error(f"Error in text analysis endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/health-advice', methods=['POST'])
def health_advice():
    """
    Endpoint for getting health-related advice using AI21
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400

        query = data['query']
        user_profile = data.get('userProfile', {})
        
        # Create a prompt that includes user profile context
        prompt = f"""Given this user profile:
{json.dumps(user_profile, indent=2)}

Please provide health advice for this query:
{query}

Focus on personalized, actionable advice that takes into account the user's profile."""

        # Get advice from AI21
        response = ai21_client.chat.create(
            model="j2-ultra",
            messages=[{"role": "user", "content": prompt}]
        )

        return jsonify({
            'advice': response.choices[0].message.content
        })

    except Exception as e:
        logger.error(f"Error in health advice endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)