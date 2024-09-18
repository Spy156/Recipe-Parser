import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import ast

class RecipeParser:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def train_from_csv(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['ingredients_str'] = self.df['ingredients'].apply(lambda x: ' '.join(ast.literal_eval(x)))
        
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['ingredients_str'])

    def save_model(self, model_path='recipe_model.joblib'):
        joblib.dump({
            'df': self.df,
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }, model_path)

    def load_model(self, model_path='recipe_model.joblib'):
        model_data = joblib.load(model_path)
        self.df = model_data['df']
        self.vectorizer = model_data['vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']

    def predict(self, dish_name, num_people):
        query_vec = self.vectorizer.transform([dish_name])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-6:-1]
        
        best_match = self.df.iloc[related_docs_indices[0]]
        
        ingredients = ast.literal_eval(best_match['ingredients'])
        scaled_ingredients = [f"{ingredient.split()[0]} {float(ingredient.split()[1]) * num_people / int(best_match['n_steps']):.2f} {' '.join(ingredient.split()[2:])}" for ingredient in ingredients]
        
        return {
            'name': best_match['name'],
            'ingredients': scaled_ingredients,
            'instructions': ast.literal_eval(best_match['steps']),
            'servings': num_people
        }
