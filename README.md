# SMS Spam Detection Using Machine Learning

## Overview
A machine learning web application that classifies SMS messages as **Spam** or **Ham (Not Spam)** using NLP, TF-IDF and Naive Bayes.

## Technologies
- Python
- Flask
- Pandas, NumPy
- Scikit-learn
- NLTK
- TF-IDF
- Naive Bayes
- HTML, CSS

## Workflow
SMS Dataset → Text Preprocessing → TF-IDF Vectorization → Naive Bayes Classifier → Spam/Ham Prediction → Flask Web App

## Model Performance
The trained model achieved approximately **97% accuracy** on the test data.

Evaluation metrics include Accuracy, Precision, Recall, F1-score and Confusion Matrix.

## Project Structure
```text
SMS-Spam-Detection/
├── model/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── preprocess.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Clone the repository
```bash
git clone https://github.com/prasidhkunder/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

### Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Example
**Input:** Congratulations! You have won a free prize. Click now!

**Prediction:** Spam

## Future Improvements
- Compare multiple machine learning algorithms.
- Add prediction confidence scores.
- Improve NLP preprocessing.
- Improve the user interface.
- Deploy the application online.

## Author
**Prasidh Kunder**  
M.Tech Student | Python | Machine Learning | AI

GitHub: https://github.com/prasidhkunder
