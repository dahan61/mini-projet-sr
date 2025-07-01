import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the reshaped data
combined_features = pd.read_csv('final_processed_reviews_data.csv')

# Step 1: Aggregate Ratings for Each Restaurant (item_id)
rating_columns = [
    'atmosphere',
    'food',
    'location_rating',
    'rooms',
    'service_rating'
]

# Aggregate (calculate mean) ratings for each restaurant (item_id)
aggregated_ratings = combined_features.groupby('item_id')[rating_columns].mean()

# Step 2: Normalize the ratings (standardize them)
scaler = StandardScaler()
aggregated_ratings_scaled = scaler.fit_transform(aggregated_ratings)

# Step 3: Compute Cosine Similarity Between Restaurants
cosine_sim = cosine_similarity(aggregated_ratings_scaled)

# Step 4: Recommend Similar Restaurants
def recommend_restaurants(restaurant_id, cosine_sim, top_n=5):
    # Get the index of the restaurant
    idx = aggregated_ratings.index.get_loc(restaurant_id)
    
    # Get the pairwise similarity scores for the given restaurant
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the restaurants based on similarity scores (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N most similar restaurants (excluding the restaurant itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the indices of the most similar restaurants
    restaurant_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar restaurants
    recommended_restaurants = aggregated_ratings.iloc[restaurant_indices]
    return recommended_restaurants[['atmosphere', 'food', 'location_rating', 'rooms', 'service_rating']]

# Streamlit app UI
st.title("Restaurant Recommendation System")
st.write("This system recommends similar restaurants based on ratings in various categories.")

# Input for restaurant item_id
restaurant_id = st.text_input("Enter the restaurant name or ID:", "مقهى غيث Gaith coffee")

# Display recommendations
if restaurant_id:
    try:
        recommended_restaurants = recommend_restaurants(restaurant_id, cosine_sim, top_n=5)
        st.write(f"Recommended restaurants based on {restaurant_id}:")
        st.dataframe(recommended_restaurants)
    except Exception as e:
        st.error(f"Error: {str(e)}")
