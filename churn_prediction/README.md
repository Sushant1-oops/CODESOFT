# Movie Genre Classification

## 🎯 Objective
To build a machine learning model that classifies movie genres based on textual data using natural language processing (NLP) and supervised learning techniques.

## 📊 Dataset
A dataset containing movie descriptions and their corresponding genres. The dataset is preprocessed and used for training and evaluating various ML models.

## 🛠️ Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Regular Expressions (`re`)
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM (LinearSVC)
- Joblib (for saving the model)

## 🧪 Models Used
- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Linear SVM**

Model evaluation was done using precision, recall, F1-score via `classification_report`.

## 🧹 Preprocessing Steps
- Lowercasing the text
- Removing special characters using regex
- TF-IDF vectorization
- Label encoding of the genre labels

## 📝 How to Run

1. Clone the repository or download the project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt

Open and run the notebook:
jupyter notebook movie_genre_classification.ipynb

