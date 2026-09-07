import pandas as pd
import joblib
import nltk

from preprocess import preprocess

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")

print("Loading dataset...")

df = pd.read_csv("spam.csv", encoding="latin1")

df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print("Dataset loaded successfully!")
print(df.head())

print("\nPreprocessing text...")

df["text"] = df["text"].apply(preprocess)

print("Text preprocessing completed!")

X = df["text"]
y = df["label"]

print("\nConverting text into TF-IDF vectors...")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

print("Vectorization completed!")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Naive Bayes model...")

model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed!")

print("\nEvaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("==============================\n")

print(classification_report(y_test, predictions))

print("\nSaving model...")

joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved successfully!")
print("Vectorizer saved successfully!")

print("\nTraining Completed Successfully!")