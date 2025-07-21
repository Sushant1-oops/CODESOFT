# Movie Genre Classification

## Overview
This project is focused on classifying movie genres using Natural Language Processing (NLP) techniques and machine learning models. It takes a movie title and plot description as input and predicts the most likely genre from a predefined set.

The project includes both a training pipeline and a deployed web app built with Streamlit.

---

## Project Structure

Movie_genre_classification/
│
├── Movie_genre.ipynb 
├── app.py
├── genre_model.pkl
├── movie_vectorizer.pkl 
├── requirements.txt
└── README.md 
---

## Features

- Text preprocessing with regular expressions
- TF-IDF vectorization for feature extraction
- Label encoding of genres
- Model training using Logistic Regression
- Model saved using `joblib`
- Interactive web interface using Streamlit

---

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- joblib
- Streamlit

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sushant1-oops/CODESOFT.git
cd CODESOFT/Movie_genre_classification

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run app.py

## Model Performance

The model was evaluated using `classification_report` from scikit-learn on the test dataset. Below is the classification report:
           precision    recall  f1-score   support

    Drama       0.87      0.91      0.89       140
   Comedy       0.82      0.79      0.80       105
 Thriller       0.81      0.77      0.79        80
  Romance       0.83      0.84      0.84        75
   Action       0.86      0.83      0.84        68
   Horror       0.78      0.76      0.77        60
    Other       0.75      0.73      0.74        55

accuracy                           0.84       583



