import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from recipe_parser import RecipeParser
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

recipe_parser = RecipeParser()

# Load the model
try:
    food_model = tf.keras.models.load_model('food_recognition_model.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Load class names
try:
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f]
    logging.info(f"Loaded {len(class_names)} class names")
except Exception as e:
    logging.error(f"Error loading class names: {str(e)}")
    raise

# Load or train recipe model
if os.path.exists('recipe_model.joblib'):
    recipe_parser.load_model()
    logging.info("Recipe model loaded successfully")
else:
    logging.warning("Recipe model not found, training new model")
    recipe_parser.train_from_csv('RAW_recipes.csv')
    recipe_parser.save_model()

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        logging.warning("No image file provided in request")
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    logging.info(f"Received image: {file.filename}")
    
    try:
        # Read the file into a BytesIO object
        img_bytes = io.BytesIO(file.read())
        
        # Open the image using PIL
        img = Image.open(img_bytes)
        logging.info(f"Image opened successfully, format: {img.format}, size: {img.size}")
        
        # Resize the image
        img = img.resize((224, 224))
        logging.info("Image resized to (224, 224)")
        
        # Convert to array and preprocess
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        logging.info("Making prediction")
        predictions = food_model.predict(img_array)
        logging.info("Prediction completed")
        
        top_3_indices = predictions[0].argsort()[-3:][::-1]
        
        result = [
            {'class': class_names[i], 'probability': float(predictions[0][i])}
            for i in top_3_indices
        ]
        
        logging.info(f"Top prediction: {result[0]['class']} with probability {result[0]['probability']}")

        # Get recipe for the top prediction
        top_dish = class_names[top_3_indices[0]]
        recipe = recipe_parser.predict(top_dish, 4)
        
        if recipe is None:
            logging.warning(f"Could not generate recipe for {top_dish}")
            recipe = {"name": top_dish, "ingredients": [], "instructions": [], "servings": 4}
        else:
            logging.info(f"Recipe generated for {top_dish}")

        return jsonify({
            'predictions': result,
            'recipe': recipe
        })
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
