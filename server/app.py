from flask import Flask, request, jsonify
from flask_cors import CORS
from recipe_parser import RecipeParser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
CORS(app)

recipe_parser = RecipeParser()
food_model = load_model('food_classification_model.h5')


with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f]

if os.path.exists('recipe_model.joblib'):
    recipe_parser.load_model()
else:
    recipe_parser.train_from_csv('RAW_recipes.csv')
    recipe_parser.save_model()

@app.route('/predict', methods=['POST'])
def predict_recipe():
    data = request.json
    dish_name = data.get('dish_name', '')
    num_people = data.get('num_people', 1)
    
    recipe = recipe_parser.predict(dish_name, num_people)
    
    return jsonify(recipe)

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    img = load_img(file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    predictions = food_model.predict(img_array)
    top_3_indices = predictions[0].argsort()[-3:][::-1]
    
    result = [
        {'class': class_names[i], 'probability': float(predictions[0][i])}
        for i in top_3_indices
    ]

    #
    top_dish = class_names[top_3_indices[0]]
    recipe = recipe_parser.predict(top_dish, 4) 

    return jsonify({
        'predictions': result,
        'recipe': recipe
    })

if __name__ == '__main__':
    app.run(debug=True)