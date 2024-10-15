#Making movie recommendation system using ML with Python:


# loading the data using pandas
import pandas as pd
# Load movies and ratings data
movies = pd.read_csv('movies.csv')  # Adjust path as necessary
ratings = pd.read_csv('ratings.csv')  # Adjust path as necessary

# merging the data
data = pd.merge(ratings, movies, on='movieId')

#creating a user item matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# bulding  the recommendation engine 
#1-calculating the simialrities :

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Fill NaN values with 0s and compute cosine similarity
similarity_matrix = cosine_similarity(user_movie_matrix.fillna(0))

#2-create the recommadation function :

def get_recommendations(movie_title):
    if movie_title not in user_movie_matrix.columns:
        return "Movie not found."
    
    idx = user_movie_matrix.columns.get_loc(movie_title)
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar movies
    top_movies_indices = [i[0] for i in sim_scores[1:6]]
    return user_movie_matrix.columns[top_movies_indices].tolist()

    # final testih of movie recommadation function :
    print(get_recommendations('The Matrix'))  # Replace with any movie title from your dataset

