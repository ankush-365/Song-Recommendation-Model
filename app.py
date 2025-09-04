import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import os

# Set page configuration
st.set_page_config(
    page_title="Song Recommendation Model",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .input-section {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .song-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .artist-name {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .similarity-score {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .selected-song {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .stTextInput > div > div > input {
        background-color: #2b2b2b;
        border: 2px solid #444;
        border-radius: 8px;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    
    .centered-content {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load the recommendation artifacts with caching for performance."""
    try:
        # Load combined vectors
        if os.path.exists('combined_vec.pkl'):
            combined_vec = joblib.load('combined_vec.pkl')
        elif os.path.exists('combined_vec.joblib'):
            combined_vec = joblib.load('combined_vec.joblib')
        else:
            st.error("❌ Combined vectors file not found. Please ensure 'combined_vec.pkl' or 'combined_vec.joblib' exists.")
            return None, None, None
        
        # Cast to float32 for memory efficiency
        combined_vec = combined_vec.astype(np.float32)
        
        # Load songs dataframe
        if os.path.exists('songs_df.parquet'):
            songs_df = pd.read_parquet('songs_df.parquet')
        elif os.path.exists('songs_df.pkl'):
            songs_df = joblib.load('songs_df.pkl')
        else:
            st.error("❌ Songs dataframe not found. Please ensure 'songs_df.parquet' or 'songs_df.pkl' exists.")
            return None, None, None
        
        # Try to load pre-trained nearest neighbors model (optional)
        nn_model = None
        if os.path.exists('nn_model.pkl'):
            try:
                nn_model = joblib.load('nn_model.pkl')
            except Exception as e:
                pass
        elif os.path.exists('nn_model.joblib'):
            try:
                nn_model = joblib.load('nn_model.joblib')
            except Exception as e:
                pass
        
        # Validate data consistency
        if len(combined_vec) != len(songs_df):
            st.error(f"❌ Data mismatch: {len(combined_vec)} vectors vs {len(songs_df)} songs")
            return None, None, None
        
        return combined_vec, songs_df, nn_model
        
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        return None, None, None

def find_song_index(song_name, songs_df):
    """Find song index by name (case-insensitive search)."""
    # First, try exact match (case-insensitive)
    mask = songs_df['song_name'].str.lower() == song_name.lower()
    matches = songs_df[mask]
    
    if len(matches) > 0:
        return matches.index[0], None
    
    # If no exact match, try partial match
    mask = songs_df['song_name'].str.lower().str.contains(song_name.lower(), na=False)
    matches = songs_df[mask]
    
    if len(matches) > 0:
        return matches.index[0], matches
    
    return None, None

def get_recommendations_nn(query_idx, nn_model, combined_vec, n_recommendations=10):
    """Get recommendations using pre-trained NearestNeighbors model."""
    distances, indices = nn_model.kneighbors([combined_vec[query_idx]], n_neighbors=n_recommendations + 1)
    rec_indices = indices[0][1:]  # Skip first element (query song itself)
    similarities = 1 - distances[0][1:]  # Convert distance to similarity
    return rec_indices, similarities

def get_recommendations_cosine(query_idx, combined_vec, n_recommendations=10):
    """Get recommendations using cosine similarity."""
    query_vector = combined_vec[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_vector, combined_vec)[0]
    
    # Get indices sorted by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Remove the query song itself and get top N
    rec_indices = []
    rec_similarities = []
    
    for idx in sorted_indices:
        if idx != query_idx:
            rec_indices.append(idx)
            rec_similarities.append(similarities[idx])
        
        if len(rec_indices) >= n_recommendations:
            break
    
    return np.array(rec_indices), np.array(rec_similarities)

def main():
    # Load artifacts
    combined_vec, songs_df, nn_model = load_artifacts()
    
    if combined_vec is None or songs_df is None:
        st.stop()
    
    # Main header
    st.markdown('<h1 class="main-header">🎵 Song Recommendation Model</h1>', unsafe_allow_html=True)
    
    # Center the content
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        song_input = st.text_input(
            "**Enter Song Name:**",
            placeholder="e.g., Shape of You, Snowman, Blinding Lights...",
            label_visibility="visible"
        )
    
    with col2:
        n_recs = st.selectbox(
            "**Recommendations:**",
            options=[5, 10, 15, 20],
            index=1
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process recommendations when song is entered
    if song_input:
        # Find the song
        query_idx, matches = find_song_index(song_input, songs_df)
        
        if query_idx is not None:
            # Display selected song
            selected_song = songs_df.loc[query_idx]
            st.markdown(f"""
            <div class="selected-song">
                🎵 Selected: {selected_song['song_name']} by {selected_song['artist']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show multiple matches if found
            if matches is not None and len(matches) > 1:
                st.info(f"Found multiple matches. Using the first one: **{selected_song['song_name']}**")
                with st.expander("📋 See all matches"):
                    st.dataframe(
                        matches[['song_name', 'artist']].reset_index(drop=True),
                        use_container_width=True
                    )
            
            try:
                # Generate recommendations
                if nn_model is not None:
                    rec_indices, similarities = get_recommendations_nn(query_idx, nn_model, combined_vec, n_recs)
                    method_used = "Nearest Neighbors"
                else:
                    rec_indices, similarities = get_recommendations_cosine(query_idx, combined_vec, n_recs)
                    method_used = "Cosine Similarity"
                
                # Display card-style recommendations
                st.markdown("### 🎯 **Top Recommendations**")
                
                # Create cards for top 3 recommendations
                for i, (idx, sim) in enumerate(zip(rec_indices[:3], similarities[:3])):
                    rec_song = songs_df.loc[idx]
                    
                    recommendation_html = f"""
                    <div class="recommendation-card">
                        <div class="song-title">#{i+1} {rec_song['song_name']}</div>
                        <div class="artist-name">by {rec_song['artist']}</div>
                        <div class="similarity-score">Similarity: {sim:.4f}</div>
                    </div>
                    """
                    st.markdown(recommendation_html, unsafe_allow_html=True)
                
                # Comprehensive results table
                st.markdown("---")
                st.markdown(f"### 📊 **Complete Results** ({method_used})")
                
                # Prepare results for table
                recommendations = []
                for i, (idx, sim) in enumerate(zip(rec_indices, similarities)):
                    rec_song = songs_df.loc[idx]
                    recommendations.append({
                        'Rank': i + 1,
                        'Song Name': rec_song['song_name'],
                        'Artist': rec_song['artist'],
                        'Similarity Score': f"{sim:.4f}"
                    })
                
                # Create and display the dataframe
                rec_df = pd.DataFrame(recommendations)
                
                st.dataframe(
                    rec_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn(
                            "Rank",
                            help="Recommendation ranking",
                            format="%d"
                        ),
                        "Song Name": st.column_config.TextColumn(
                            "Song Name",
                            help="Recommended song title",
                            width="large"
                        ),
                        "Artist": st.column_config.TextColumn(
                            "Artist",
                            help="Artist name",
                            width="medium"
                        ),
                        "Similarity Score": st.column_config.TextColumn(
                            "Similarity Score",
                            help="How similar this song is to your query (higher = more similar)",
                            width="small"
                        )
                    }
                )
                
                # Statistics
                avg_similarity = np.mean(similarities)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Average Similarity", f"{avg_similarity:.4f}")
                with col2:
                    st.metric("🔥 Best Match", f"{similarities[0]:.4f}")
                with col3:
                    st.metric("📊 Method Used", method_used)
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")
        
        else:
            st.warning(f"⚠️ Couldn't find '{song_input}' in the database. Please try a different song name.")
            
            # Show some suggestions
            st.markdown("**💡 Here are some popular songs in the database:**")
            sample_songs = songs_df.sample(min(10, len(songs_df)))[['song_name', 'artist']]
            st.dataframe(sample_songs.reset_index(drop=True), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🎵 Hybrid Recommender combines lyrics embeddings and audio features for better music discovery</p>
        <p>Built with Streamlit | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()