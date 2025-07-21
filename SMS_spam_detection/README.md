# SMS Spam Detection

A machine learning project that classifies SMS messages as either **Spam** or **Ham** (legitimate) using Support Vector Machine (SVM) with TF-IDF vectorization.

##  Overview

This project implements a text classification system to automatically detect spam SMS messages. The model is trained on a dataset of labeled SMS messages and can predict whether a new message is spam or legitimate with high accuracy.

##  Features

- **High Accuracy**: Achieves 98% accuracy on test data
- **Real-time Classification**: Interactive web interface for instant spam detection
- **Lightweight Model**: Uses Linear SVM for efficient predictions
- **Easy to Use**: Simple Streamlit web application interface

##  Model Performance

| Metric | Ham (0) | Spam (1) | Overall |
|--------|---------|----------|---------|
| Precision | 98% | 98% | 98% |
| Recall | 100% | 87% | 98% |
| F1-Score | 99% | 92% | 98% |

##  Technology Stack

- **Python 3.x**
- **Machine Learning**: scikit-learn
- **Text Processing**: TF-IDF Vectorization
- **Web Framework**: Streamlit
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

##  Project Structure

```
sms-spam-detection/
├── SMS_SPAM.ipynb          # Jupyter notebook with model training
├── sms_app.py              # Streamlit web application
├── sms_model.pkl           # Trained SVM model
├── vectorizer.pkl          # TF-IDF vectorizer
├── spam.csv               # Dataset (not included in repo)
└── README.md              # Project documentation
```

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. **Install required packages**
   ```bash
   pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
   ```

3. **Download the dataset**
   - Download the SMS Spam Collection dataset
   - Save it as `spam.csv` in the project directory
   - Ensure the CSV has columns: `v1` (labels) and `v2` (messages)

##  Usage

### Training the Model

1. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook SMS_SPAM.ipynb
   ```

2. **Execute all cells** to:
   - Load and preprocess the data
   - Train the Linear SVM model
   - Evaluate model performance
   - Save the trained model and vectorizer

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run sms_app.py
   ```

2. **Open your browser** and navigate to the provided local URL (usually `http://localhost:8501`)

3. **Enter an SMS message** in the text area and click "Check Message" to get the prediction

##  Model Details

### Algorithm: Linear Support Vector Classifier (LinearSVC)
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 8,672 TF-IDF features extracted from text
- **Training Data**: 4,457 samples
- **Test Data**: 1,115 samples
- **Labels**: 
  - 0 = Ham (legitimate message)
  - 1 = Spam (unwanted message)

### Data Preprocessing Steps
1. Load dataset with proper encoding (`latin-1`)
2. Remove unnecessary columns
3. Rename columns to `label` and `message`
4. Map labels: `ham` → 0, `spam` → 1
5. Apply TF-IDF vectorization to convert text to numerical features
6. Split data into training and testing sets (80/20 split)

##  Results

The model demonstrates excellent performance:
- **Overall Accuracy**: 98%
- **Ham Detection**: 99% F1-score (excellent at identifying legitimate messages)
- **Spam Detection**: 92% F1-score (good at catching spam with minimal false positives)

##  Future Enhancements

- [ ] Add more advanced preprocessing (stemming, lemmatization)
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add confidence scores to predictions
- [ ] Create batch processing functionality
- [ ] Deploy to cloud platforms (Heroku, AWS, etc.)
- [ ] Add data visualization dashboard
- [ ] Implement multilingual support
