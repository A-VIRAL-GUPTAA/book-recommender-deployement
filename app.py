import pickle
import pandas as pd
from flask import Flask, jsonify, request, render_template
import os

# --- FLASK APP SETUP ---
app = Flask(__name__)

# --- GLOBAL MODEL COMPONENTS ---
# Initialize with empty/None values
popular_df = pd.DataFrame() 
pt = pd.DataFrame()        
scores = None              
all_books = []             

def load_model_components():
    """
    Loads all necessary model components from the 'recommendation_system.pkl' file.
    
    The 'global' declarations must be at the beginning to apply to the entire function scope.
    """
    global popular_df, pt, scores, all_books
    
    try:
        # Load the uploaded pickle file
        with open('recommendation_system.pkl', 'rb') as f:
            data = pickle.load(f)
            
            # --- Check the structure of the loaded data ---
            if isinstance(data, (tuple, list)) and len(data) >= 3:
                # Assuming the pickle file contains: (popular_df, pt, scores)
                popular_df, pt, scores = data[0], data[1], data[2]
            elif isinstance(data, dict):
                # Assuming the pickle file contains a dictionary with keys
                popular_df = data.get('popular_df', popular_df)
                pt = data.get('pt', pt)
                scores = data.get('scores', scores)
            else:
                # Fallback: If it's just a DataFrame
                popular_df = data if isinstance(data, pd.DataFrame) else pd.DataFrame()

            # Post-load preparation: Extract all unique book names for the dropdown
            if not pt.empty:
                all_books = pt.index.tolist()
            else:
                # If pt is not available, use unique titles from popular_df
                all_books = popular_df['Book-Title'].unique().tolist()
                
            print("--- Model components loaded successfully! ---")

    except Exception as e:
        print(f"Error loading model components: {e}")
        print("Using mock data for demonstration. Check 'recommendation_system.pkl' structure.")
        
        # --- MOCK DATA FOR FALLBACK (If loading fails) ---
        # NOTE: Removed redundant 'global' declarations here.
        popular_df = pd.DataFrame([
            {'Book-Title': f'Mock Popular Book {i}', 'Book-Author': f'Author {i}', 
             'Image-URL-M': f'https://placehold.co/80x120/4f46e5/ffffff?text=Book{i}', 
             'Num-Ratings': 1000 + i, 'Avg-Rating': 4.0 + (i / 100)}
            for i in range(1, 51)
        ])
        
        all_books = popular_df['Book-Title'].tolist()
        # --- END MOCK DATA ---


def recommend(book_name):
    """
    Performs the collaborative filtering recommendation using the loaded model.
    """
    global pt, scores, popular_df
    
    if pt.empty or scores is None:
        # If model failed to load, return random popular books as mock recommendations
        print("Model data not available. Returning random popular titles.")
        # Ensure popular_df is available, even if mocked
        if popular_df.empty:
             return ["Model data unavailable. Please check backend setup."]
        return popular_df['Book-Title'].sample(min(5, len(popular_df))).tolist()

    try:
        # 1. Get the index of the input book
        index = pt.index.get_loc(book_name)
        
        # 2. Get and sort similarity scores, skipping the book itself [1:6]
        similar_items = list(enumerate(scores[index]))
        sorted_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:6]
        
        recommendations = []
        for i, _ in sorted_items:
            # 3. Get the book title
            book_title = pt.index[i]
            recommendations.append(book_title)
            
        return recommendations

    except KeyError:
        return [f"Error: '{book_name}' not found in the trained model's index."]
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
        return ["An unexpected error occurred during recommendation."]


# --- FLASK ROUTES ---

# Uses @app.before_first_request to run once before the server handles its first request.
# This is a good way to ensure the large model files are loaded only once.
@app.before_request
def startup():
    """Load model components before the first request."""
    load_model_components()
    
@app.route('/')
def index():
    """Renders the main HTML page (the Home page)."""
    return render_template('index.html')

@app.route('/get_popular_books', methods=['GET'])
def get_popular_books_api():
    """API endpoint for the Home page to get the top 50 books."""
    
    top_books_list = popular_df.head(50).to_dict('records')
    
    return jsonify({
        'popular_books': top_books_list
    })

@app.route('/get_all_books', methods=['GET'])
def get_all_books_api():
    """API endpoint to get the list of all book titles for the dropdown."""
    return jsonify({
        'all_book_titles': all_books
    })


@app.route('/get_recommendations', methods=['POST'])
def get_recommendation_api():
    """API endpoint to get recommendations for a specific book title."""
    
    data = request.get_json()
    
    if not data or 'book_name' not in data:
        return jsonify({'error': 'Missing "book_name" in request body.'}), 400

    book_name = data['book_name']
    
    # Run the recommendation logic
    recommended_books = recommend(book_name)
    
    return jsonify({
        'recommendations': recommended_books
    })

if __name__ == '__main__':
    # For local testing, ensure the templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    app.run(debug=True)

