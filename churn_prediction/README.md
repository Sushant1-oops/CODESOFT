# Movie Genre Classification

## Overview
This project focuses on classifying movie genres using Natural Language Processing (NLP) techniques and machine learning. It takes a movie title and plot summary as input and predicts the most probable genre. The goal is to assist in organizing, tagging, and recommending films by genre using automated tools.

The model is trained using TF-IDF vectorization and Logistic Regression, and the project includes an interactive web app built with Streamlit.

---

## Technologies Used
- Python
- Pandas
- NumPy
- scikit-learn
- TfidfVectorizer
- Logistic Regression
- Joblib
- Streamlit

---

## Preprocessing Steps
- Lowercasing of text
- Removal of special characters using regex
- TF-IDF vectorization for converting text into numerical features
- Label encoding of genre names
- Train/test split (typically 80/20)
- Model training with Logistic Regression

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Sushant1-oops/CODESOFT.git
cd CODESOFT/Movie_genre_classification

**2. Install Dependencies**
pip install -r requirements.txt

**3. Run the Streamlit App**
streamlit run app.py


**Classification Report**
               precision    recall  f1-score   support

        Drama       0.87      0.91      0.89       140
       Comedy       0.82      0.79      0.80       105
     Thriller       0.81      0.77      0.79        80
      Romance       0.83      0.84      0.84        75
       Action       0.86      0.83      0.84        68
       Horror       0.78      0.76      0.77        60
        Other       0.75      0.73      0.74        55

    accuracy                           0.84       583
   macro avg       0.82      0.80      0.81       583
weighted avg       0.84      0.84      0.84       583

**Folder Structure**
Movie_genre_classification/
│
├── Movie_genre.ipynb           
├── app.py                     
├── genre_model.pkl             
├── movie_vectorizer.pkl       
├── requirements.txt            
└── README.md                   
