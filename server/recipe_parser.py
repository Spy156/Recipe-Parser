import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
import logging

class RecipeParser:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.recipes_df = None

    def train_from_csv(self, csv_file):
        logging.info(f"Training model from CSV file: {csv_file}")
        self.recipes_df = pd.read_csv(csv_file)
        
        # Check for missing values in the 'name' column
        if self.recipes_df['name'].isnull().any():
            logging.warning("Found NaN values in 'name' column. Filling with empty strings.")
            self.recipes_df['name'] = self.recipes_df['name'].fillna('')

        self.vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(self.recipes_df['name'])
        self.model = NearestNeighbors(n_neighbors=1, metric='cosine').fit(tfidf_matrix)
        logging.info("Model training completed")

    def save_model(self, filename='recipe_model.joblib'):
        logging.info(f"Saving model to {filename}")
        dump((self.model, self.vectorizer, self.recipes_df), filename)

    def load_model(self, filename='recipe_model.joblib'):
        logging.info(f"Loading model from {filename}")
        self.model, self.vectorizer, self.recipes_df = load(filename)

    def predict(self, dish_name, num_people):
        logging.info(f"Predicting recipe for dish: {dish_name}, serving {num_people} people")
        try:
            dish_vector = self.vectorizer.transform([dish_name])
            distances, indices = self.model.kneighbors(dish_vector)
            closest_recipe = self.recipes_df.iloc[indices[0][0]]
            
            ingredients = eval(closest_recipe['ingredients'])
            instructions = eval(closest_recipe['steps'])
            
            # Scale ingredients for the number of people
            scale_factor = num_people / closest_recipe['n_steps']
            scaled_ingredients = []
            for ingredient in ingredients:
                try:
                    amount, unit, name = ingredient.split(' ', 2)
                    amount = float(amount) * scale_factor
                    scaled_ingredients.append(f"{amount:.2f} {unit} {name}")
                except ValueError:
                    logging.warning(f"Could not parse ingredient: {ingredient}. Adding as is.")
                    scaled_ingredients.append(ingredient)

            recipe = {
                'name': closest_recipe['name'],
                'ingredients': scaled_ingredients,
                'instructions': instructions,
                'servings': num_people
            }
            
            return recipe
        except Exception as e:
            logging.error(f"Error in recipe prediction: {str(e)}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = RecipeParser()
    parser.train_from_csv('RAW_recipes.csv')
    parser.save_model()
    
    # Test prediction
    recipe = parser.predict("Chocolate Cake", 4)
    if recipe:
        print(f"Recipe for {recipe['name']}:")
        print("Ingredients:")
        for ingredient in recipe['ingredients']:
            print(f"- {ingredient}")
        print("\nInstructions:")
        for i, step in enumerate(recipe['instructions'], 1):
            print(f"{i}. {step}")
    else:
        print("Could not generate recipe.")
