import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")

# Clean titles (important!)
movies["title"] = movies["title"].str.lower().str.strip()

# Convert genre text to vectors
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies["genre"])

# Compute similarity
similarity = cosine_similarity(genre_matrix)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower().strip()

    if movie_name not in movies["title"].values:
        print("❌ Movie not found!")
        return

    index = movies[movies["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print("\n🎬 Recommended Movies:")
    for i in sorted_scores:
        print("👉", movies.iloc[i[0]].title.title())

# User input
movie = input("Enter a movie name: ")
recommend(movie)