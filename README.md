# 🎵 Song Recommendation System  

An AI-powered music recommendation system that suggests songs based on **lyrics similarity** and **audio features**.  
Inspired by platforms like Spotify, this project combines **Machine Learning** and **NLP techniques** to deliver meaningful recommendations.  

---

## 📌 Features  
- Recommends songs similar to a given track  
- Uses **lyrics (NLP)** and **audio features** for better accuracy  
- Interactive web app built with **Streamlit**  
- Lightweight and easy to use  

---

## 🗂️ Dataset  
The dataset contains information about songs, including:  
- **Song name**  
- **Artist name**  
- **Lyrics**  
- **Genres**  
- **Audio features** (danceability, energy, loudness, tempo, etc.)  

---

## ⚙️ Tech Stack  
- **Python** – Core programming language  
- **Pandas, NumPy** – Data handling  
- **Scikit-learn** – Cosine similarity, preprocessing  
- **NLP techniques** – Tokenization, stemming, vectorization of lyrics  
- **Streamlit** – Interactive frontend for recommendations  

---

## 🚀 How It Works  
1. Lyrics are preprocessed (lowercasing, tokenization, stemming).  
2. Both **lyrics embeddings** and **audio feature vectors** are combined.  
3. **Cosine similarity** is used to find the closest matching songs.  
4. The top recommendations are displayed on the **Streamlit app**.  

---

## 📷 Demo
🔗 Live Demo: https://song-recommendation-model.streamlit.app/
