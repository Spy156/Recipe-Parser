import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RecipeParser:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.recipes_df = None

    def train_from_csv(self, csv_file):
        try:
            logging.info(f"Loading data from CSV file: {csv_file}")
            self.recipes_df = pd.read_csv(csv_file)
            
            # Check for NaN values in the 'name' column
            nan_count = self.recipes_df['name'].isna().sum()
            if nan_count > 0:
                logging.warning(f"Found {nan_count} NaN values in 'name' column. Removing these rows.")
                self.recipes_df = self.recipes_df.dropna(subset=['name'])
            
            logging.info(f"Loaded {len(self.recipes_df)} recipes")
            
            # Fit and transform the TF-IDF vectorizer
            self.tfidf_matrix = self.vectorizer.fit_transform(self.recipes_df['name'])
            logging.info("TF-IDF matrix created successfully")
            
        except FileNotFoundError:
            logging.error(f"CSV file not found: {csv_file}")
            raise
        except pd.errors.EmptyDataError:
            logging.error(f"CSV file is empty: {csv_file}")
            raise
        except ValueError as e:
            logging.error(f"Error processing CSV data: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise

    def find_similar_recipes(self, query, n=5):
        if self.tfidf_matrix is None:
            raise ValueError("Model not trained. Call train_from_csv() first.")
        
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-n-1:-1]
        
        return self.recipes_df.iloc[related_docs_indices]

def main():
    recipe_parser = RecipeParser()
    
    try:
        recipe_parser.train_from_csv('RAW_recipes.csv')
        logging.info("Model trained successfully")
        
        # Example usage
        query = "chicken soup"
        similar_recipes = recipe_parser.find_similar_recipes(query)
        print(f"Recipes similar to '{query}':")
        print(similar_recipes[['name', 'ingredients']])
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()