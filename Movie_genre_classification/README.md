# Movie Genre Classification 🎬

## Overview
This project is focused on classifying movie genres using Natural Language Processing (NLP) techniques and machine learning models. It takes a movie title and plot description as input and predicts the most likely genre from a predefined set.

The project includes both a training pipeline and a deployed web app built with Streamlit.

---

## Project Structure

Movie_genre_classification/
│
├── Movie_genre.ipynb # Jupyter notebook for data preprocessing, training, and evaluation
├── app.py # Streamlit app for live predictions
├── genre_model.pkl # Saved ML model
├── movie_vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Features

- Text preprocessing (lowercasing, cleaning, tokenization)
- Feature extraction using **TF-IDF**
- Multi-class classification (27 genres)
- Model training using **Logistic Regression**
- Hyperparameter tuning using **GridSearchCV**
- Deployment-ready web interface using **Streamlit**

---

## Model Details

- **Algorithm**: Logistic Regression (One-vs-Rest strategy)
- **Vectorizer**: TF-IDF (top 5000 features)
- **Hyperparameters**:
  - `C = 10`
  - `solver = 'liblinear'`
  - `class_weight = 'balanced'`
  - `max_iter = 1000`

---

## Model Performance

Tested on a dataset of 54,214 movie plots:

- **Accuracy**: 76%
- **Macro F1-score**: 0.79
- **Weighted F1-score**: 0.76

The model performs well across both common and rare genres thanks to balanced class weights and TF-IDF representation.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Sushant1-oops/CODESOFT.git
cd CODESOFT/Movie_genre_classification

### 2. Install dependencies
```
pip install -r requirements.txt

### 3. Run the app
```
streamlit run app.py

## Project Structure
```
Movie_genre_classification/
├── Movie_genre.ipynb       
├── app.py                  
├── genre_model.pkl        
├── movie_vectorizer.pkl    
├── requirements.txt        
└── README.md               

