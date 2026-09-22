"""
Example script showing how to use the MovieRecommender class
Run this after you have movies_metadata.csv in the same folder
"""

from movie_recommender import MovieRecommender

# Initialize the recommender (this takes a minute on first run)
print("Loading recommender system...")
recommender = MovieRecommender('movies_metadata.csv')

# Example 1: Get recommendations
print("\n" + "="*50)
print("EXAMPLE 1: Basic Recommendations")
print("="*50)

movies_to_try = ['Toy Story', 'Inception', 'The Dark Knight']

for movie in movies_to_try:
    recommendations = recommender.recommend(movie, n=5)
    print(f"\nIf you like '{movie.title()}', try:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec.title()}")

# Example 2: Error handling
print("\n" + "="*50)
print("EXAMPLE 2: Movie Not Found")
print("="*50)

result = recommender.recommend('Random Fake Movie Title 12345', n=5)
print(f"Result: {result}")

# Example 3: Different number of recommendations
print("\n" + "="*50)
print("EXAMPLE 3: Get 10 Recommendations")
print("="*50)

recommendations = recommender.recommend('The Matrix', n=10)
print("\nTop 10 recommendations for 'The Matrix':")
for i, movie in enumerate(recommendations, 1):
    print(f"{i:2}. {movie.title()}")

# Example 4: Interactive mode
print("\n" + "="*50)
print("EXAMPLE 4: Try It Yourself")
print("="*50)

try:
    user_movie = input("\nEnter a movie title (or press Enter to skip): ").strip()
    if user_movie:
        recs = recommender.recommend(user_movie, n=5)
        if isinstance(recs, str):  # Error message
            print(recs)
        else:
            print(f"\nBased on '{user_movie}':")
            for i, movie in enumerate(recs, 1):
                print(f"{i}. {movie.title()}")
except KeyboardInterrupt:
    print("\n(Cancelled)")

print("\nDone!")
