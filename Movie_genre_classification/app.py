import joblib
import streamlit as st
model=joblib.load("genre_model.pkl")
vectorizer=joblib.load("movie_vectorizer.pkl")
st.set_page_config("MOVIE GENRE CLASSIFICATION")
st.header("MOVIE GENRE CLASSIFICATION")
input=st.text_area("Enter the movie name and title")
genre_label = {
    0: 'Action', 1: 'Adventure', 2: 'Animation', 3: 'Biography',
    4: 'Crime', 5: 'Comedy', 6: 'Documentary', 7: 'Drama',
    8: 'Family', 9: 'Fantasy', 10: 'Film-Noir', 11: 'Game-Show',
    12: 'History', 13: 'Horror', 14: 'Music', 15: 'Musical',
    16: 'Mystery', 17: 'News', 18: 'Reality-TV', 19: 'Romance',
    20: 'Sci-Fi', 21: 'Short', 22: 'Sport', 23: 'Talk-Show',
    24: 'Thriller', 25: 'War', 26: 'Western'
}
st.markdown("Enter the **movie title and plot** to predict the genre.")
if st.button("predict"):
    if input.strip()==" ":
        st.warn("Please Enter the movie name or title")
    else:
        X=vectorizer.transform([input])
        prediction=model.predict(X)[0]
        genre=genre_label.get(prediction, "unknown")
        st.success(f"predicted Genre {genre}")

