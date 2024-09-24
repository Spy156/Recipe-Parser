import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import logging

class RecipeParser:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.recipes_df = None
        self.tfidf_matrix = None

    def train_from_csv(self, csv_file):
        logging.info(f"Training model from CSV file: {csv_file}")
        try:
            # Load the CSV file
            self.recipes_df = pd.read_csv(csv_file)

            # Clean the 'name' column
            self.recipes_df['name'] = self.recipes_df['name'].fillna('Unknown Recipe')
            self.recipes_df['name'] = self.recipes_df['name'].astype(str)

            # Remove any rows with empty names after cleaning
            self.recipes_df = self.recipes_df[self.recipes_df['name'] != '']

            # Fit and transform the TF-IDF vectorizer
            self.tfidf_matrix = self.vectorizer.fit_transform(self.recipes_df['name'])
            
            logging.info(f"Model trained successfully. Processed {len(self.recipes_df)} recipes.")
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def predict(self, recipe_name, num_ingredients=4):
        if self.tfidf_matrix is None:
            logging.error("Model not trained. Please train the model first.")
            return None

        try:
            # Transform the input recipe name
            recipe_vec = self.vectorizer.transform([recipe_name])

            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(recipe_vec, self.tfidf_matrix).flatten()

            # Get the index of the most similar recipe
            similar_recipe_idx = cosine_similarities.argmax()

            # Get the similar recipe details
            similar_recipe = self.recipes_df.iloc[similar_recipe_idx]

            # Extract ingredients (assuming 'ingredients' column exists)
            if 'ingredients' in similar_recipe:
                ingredients = eval(similar_recipe['ingredients'])[:num_ingredients]
            else:
                ingredients = []

            # Extract instructions (assuming 'instructions' column exists)
            if 'instructions' in similar_recipe:
                instructions = eval(similar_recipe['instructions'])
            else:
                instructions = []

            return {
                "name": similar_recipe['name'],
                "ingredients": ingredients,
                "instructions": instructions,
                "servings": similar_recipe.get('servings', 4)  # Default to 4 if 'servings' column doesn't exist
            }
        except Exception as e:
            logging.error(f"Error predicting recipe: {str(e)}")
            return None

    def save_model(self, filename='recipe_model.joblib'):
        if self.tfidf_matrix is None:
            logging.error("Model not trained. Cannot save.")
            return

        try:
            dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'recipes_df': self.recipes_df
            }, filename)
            logging.info(f"Model saved successfully to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")

    def load_model(self, filename='recipe_model.joblib'):
        try:
            loaded_model = load(filename)
            self.vectorizer = loaded_model['vectorizer']
            self.tfidf_matrix = loaded_model['tfidf_matrix']
            self.recipes_df = loaded_model['recipes_df']
            logging.info(f"Model loaded successfully from {filename}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise