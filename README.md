# Movie Recommendation System

A machine learning-powered recommendation engine that analyzes movie metadata to suggest similar titles. Designed to demonstrate end-to-end data science capabilities including data preprocessing, feature engineering, and scalable recommendation algorithms.

---

## 🎯 Problem & Solution

**Problem:** Large movie datasets contain unstructured information (plot descriptions, genres, metadata) that's difficult to leverage for personalized recommendations.

**Solution:** Built a content-based filtering system using TF-IDF vectorization and cosine similarity to surface relevant recommendations from 45K+ movies with <100ms latency.

---

## 📊 Technical Implementation

### Architecture
- **Input:** Movie metadata (plot, genres, production details) + user query
- **Processing:** Text preprocessing → TF-IDF feature extraction → Cosine similarity ranking
- **Output:** Ranked list of recommendations

### Key Features
- **Text Processing:** NLTK-based tokenization, lemmatization, stopword removal
- **Feature Engineering:** TF-IDF with bigrams (100K features, 1.7M non-zero elements)
- **Similarity Metric:** Cosine similarity on sparse matrices
- **Dataset:** 45,466 movies × 24 attributes (~8.3MB in memory)

### Technologies
```
Python 3.9+ | Pandas | Scikit-learn | NLTK | NumPy
```

---

## 💼 Business Value

**Scalability:**
- Handles 45K+ items efficiently using sparse matrix operations
- Model trained and ready for production inference

**Accuracy:**
- Content-based approach (no cold-start problem for new movies)
- Meaningful recommendations validated on diverse genres

**Deployment Ready:**
- Modular class-based architecture
- Serializable models (pickle support)
- Clean separation of data pipeline and inference

---

## 🚀 Usage

### Quick Start
```bash
pip install -r requirements.txt
python movie_recommender.py
```

### Production Integration
```python
from movie_recommender import MovieRecommender

# Initialize once
recommender = MovieRecommender('movies_metadata.csv')

# Get recommendations (can be called repeatedly)
recommendations = recommender.recommend('Inception', n=10)
# Returns: ['Interstellar', 'The Prestige', 'Memento', ...]
```

### Persistence
```python
# Save trained model for deployment
recommender.save_model('model.pkl')

# Load in production
recommender.load_model('model.pkl')
recommender.recommend('The Dark Knight')
```

---

## 📁 Project Structure

```
.
├── movie_recommender.py      # Core ML pipeline (modular, production-ready)
├── example_usage.py          # Demonstrates various use cases
├── requirements.txt          # Dependencies
├── README.md                 # User guide
├── .gitignore               # Version control hygiene
└── movies_metadata.csv      # Data source (45K movies, 24 features)
```

---

## 🔧 Code Quality

- **Object-Oriented Design:** Encapsulated recommender logic in `MovieRecommender` class
- **Error Handling:** Graceful handling of invalid queries
- **Documentation:** Docstrings for all methods
- **Reproducibility:** Deterministic results with fixed random states
- **Extensibility:** Easy to add collaborative filtering, neural embedding models

---

## 📈 Skills Demonstrated

✅ **Data Science:** Feature engineering, text preprocessing, similarity metrics  
✅ **Software Engineering:** OOP, class design, model serialization  
✅ **Python:** Pandas, scikit-learn, NLTK, pickle  
✅ **Problem-Solving:** Translating unstructured data into actionable insights  
✅ **Production Mindset:** Scalable, deployable code (not just notebooks)  

---

## 🔮 Future Enhancements

- [ ] Hybrid approach: combine content-based + collaborative filtering
- [ ] Deep learning embeddings (Word2Vec, embeddings layer)
- [ ] User interaction tracking for personalization
- [ ] REST API for production deployment
- [ ] A/B testing framework for recommendation quality

---

## 📝 Dataset

Source: [The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

**Features Used:**
- `overview` (plot description)
- `genres` (movie categories)
- `title` (movie name)
- `vote_average`, `revenue`, `budget` (optional enrichment)

---

## 🛠️ Installation & Deployment

```bash
# 1. Clone repository
git clone https://github.com/Partha2612/movie-recommender.git
cd movie-recommender

# 2. Set up environment
pip install -r requirements.txt

# 3. Add data
# Download movies_metadata.csv from Kaggle and place in project root

# 4. Test
python example_usage.py
```

---

## 📬 Contact

For questions about this project or to discuss ML/analytics roles:
- **Email:** parthamukh26@gmail.com
- **LinkedIn:** [Your LinkedIn]
- **Portfolio:** [Your Portfolio]

---

**Built as part of MBA curriculum in Business Analytics** | Currently seeking roles in Fintech & Data Science
