import pandas as pd

# Load the dataset
df = pd.read_csv('news.csv')

# Show the shape (rows, columns)
print("✅ Dataset shape:", df.shape)

# View first 5 rows
print("\n🔍 Sample data:")
print(df.head())

# View column names
print("\n📋 Columns in the dataset:")
print(df.columns)

# Check for missing values
print("\n❓ Missing values:")
print(df.isnull().sum())

# FIRST UPDATED CODE FOR ACCURACY

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Drop the 'idd' column and any missing titles (optional)
df = df.drop(columns=['idd'])
df = df.dropna(subset=['title'])  # just 1 missing, it's safe

# Use the text column for input and label column for target
X = df['text']
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print("🧾 Confusion Matrix:")
print(cm)


#SECOND UPDATED CODE TO ENSURE OUTPUT


# Test with your own news text
custom_news = input("\n🧪 Enter news text to test: ")
custom_vector = vectorizer.transform([custom_news])
prediction = model.predict(custom_vector)
print(f"🔍 Prediction: {prediction[0]}")


#THIRD UPDATED CODE TO LOAD AND SAVE THE MODEL THEN WE WON'T BE RETRIVE NEXT TIME.


import joblib

# Save the model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Model and vectorizer saved.")
