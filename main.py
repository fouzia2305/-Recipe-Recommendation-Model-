from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app= Flask(__name__)

# Load Dataset
df = pd.read_csv(r"C:\Users\FOUZIA KOUSER\OneDrive\Desktop\cleaned_recipes.csv")

# Vectorize ingredients
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','), binary=False)
ingredient_vectors = vectorizer.fit_transform(df['ingredients_clean'])

# Compute similarity matrix
cosine_sim_matrix = cosine_similarity(ingredient_vectors)

#Function to recommend recipes
def recommend_recipes(recipe_name,num_recommendations=5,cuisine_filter=None):
    try:
        recipe_index=df[df['recipe_name'].str.lower() == recipe_name].index[0]
        sim_scores = list(enumerate(cosine_sim_matrix[recipe_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1],reverse=True)
        recommended_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]

        if cuisine_filter:
            recommended_indices = [
            idx for idx in recommended_indices
            if df.iloc[idx]['cuisine_label'] == cuisine_filter
        ]

        recommendations = df.iloc[recommended_indices][['recipe_name', 'cuisine_label']]
        return recommendations.to_dict(orient='records')
    except IndexError:
        return []

# API Route for rendering HTML page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# API Route for handling recommendation requests
@app.route('/recommend', methods=['GET'])
def recommend():
    recipe_name = request.args.get('recipe_name', '').strip().lower()
    num_recommendations = int(request.args.get('num_recommendations', 5))
    cuisine_filter = request.args.get('cuisine_filter', None)
    recommendations = []

    if not recipe_name:
        return render_template('index.html', error="Please provide a recipe name")

    recommendations = recommend_recipes(recipe_name,num_recommendations,cuisine_filter)

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)