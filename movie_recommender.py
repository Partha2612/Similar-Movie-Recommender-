import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

class MovieRecommender:
    def __init__(self, data_path='movies_metadata.csv'):
        """Initialize the recommender system"""
        self.df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.indices = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        if os.path.exists(data_path):
            self.load_and_prepare_data(data_path)
    
    def preprocess_text(self, text):
        """Clean and process text"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    
    def load_and_prepare_data(self, data_path):
        """Load CSV and prepare the data"""
        print("Loading data...")
        self.df = pd.read_csv(data_path, low_memory=False)
        
        # Create tag column from overview and genres
        self.df['tag'] = self.df['overview'].fillna('') + ' ' + self.df['genres'].fillna('')
        
        # Preprocess tags
        print("Processing text...")
        self.df['tag'] = self.df['tag'].apply(self.preprocess_text)
        self.df['title'] = self.df['title'].str.lower()
        
        # Create title index for quick lookup
        self.indices = pd.Series(
            self.df.index,
            index=self.df['title'].str.lower().str.strip()
        ).groupby(level=0).first()
        
        # Build TF-IDF matrix
        print("Building TF-IDF model...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['tag'])
        print("Model ready!")
    
    def recommend(self, title, n=10):
        """Get movie recommendations"""
        title = title.lower().strip()
        
        if title not in self.indices:
            return f"Movie '{title}' not found. Try a different title."
        
        idx = self.indices[title]
        
        cosine_scores = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()
        
        similar_movies_indices = cosine_scores.argsort()[::-1][1:n+1]
        recommendations = self.df['title'].iloc[similar_movies_indices].to_list()
        
        return recommendations
    
    def save_model(self, model_path='recommender_model.pkl'):
        """Save trained model"""
        model_data = {
            'tfidf_matrix': self.tfidf_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'df': self.df,
            'indices': self.indices
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='recommender_model.pkl'):
        """Load pre-trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.df = model_data['df']
        self.indices = model_data['indices']
        print("Model loaded successfully!")


if __name__ == "__main__":
    # Example usage
    recommender = MovieRecommender('movies_metadata.csv')
    
    # Get recommendations
    recommendations = recommender.recommend('Toy Story', n=10)
    print("\nRecommendations for 'Toy Story':")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")
    
    # Save the model for later use
    # recommender.save_model()
