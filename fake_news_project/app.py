# import streamlit as st
# import joblib

# # Load the model and vectorizer
# model = joblib.load('fake_news_model.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')

# st.title("📰 Fake News Detector")
# st.write("Enter a news article or sentence below, and the model will predict whether it's REAL or FAKE.")

# # Text input from the user
# news_text = st.text_area("Paste the news content here:")

# if st.button("Predict"):
#     if not news_text.strip():
#         st.warning("⚠️ Please enter some text first.")
#     else:
#         # Vectorize the input and make prediction
#         input_vector = vectorizer.transform([news_text])
#         prediction = model.predict(input_vector)[0]

#         if prediction == "FAKE":
#             st.error("🚨 This news is likely **FAKE**.")
#         else:
#             st.success("✅ This news is likely **REAL**.")

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit page config
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️", layout="centered")

# Main Title
st.title("📰 Fake News Detection ")
st.markdown("🚀 **Detect whether a news article is REAL or FAKE using Machine Learning**")

# Sidebar
st.sidebar.title("⚙️ About Project")
st.sidebar.info(
    """
    This app uses **TF-IDF Vectorization** + **Passive Aggressive Classifier**  
    to classify news articles as **REAL** or **FAKE**.  

    👨‍💻 Built with Python, Scikit-learn, and Streamlit.
    """
)

# ✅ Model performance info (hardcoded from your training results)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")
st.sidebar.write("**Accuracy:** 94.23%")
st.sidebar.write("**Confusion Matrix:**")
st.sidebar.write(
    np.array([[430, 20],
              [33, 436]])
)

# Tabs: One for single text, one for CSV upload
tab1, tab2 = st.tabs(["✍️ Single Article", "📂 Upload CSV"])

# ========================
# 🔹 Tab 1: Single text input
# ========================
with tab1:
    user_input = st.text_area("upload a news article here:", height=200)

    if st.button("🔍 Check News", key="check_news"):
        if user_input.strip() != "":
            # Transform input
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]

            # Show result
            if prediction == "FAKE":
                st.error("🚨 This news is **FAKE** ❌")
            else:
                st.success("✅ This news is **REAL** 🟢")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# ========================
# 🔹 Tab 2: Upload CSV
# ========================
with tab2:
    uploaded_file = st.file_uploader("📂 Upload a CSV file with a column named `text`", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("❌ The CSV must contain a column named **text**.")
            else:
                st.write("✅ File uploaded successfully!")
                st.write("📋 First 5 rows of your data:")
                st.dataframe(df.head())

                # Transform text column
                X = vectorizer.transform(df["text"].astype(str))
                predictions = model.predict(X)

                df["prediction"] = predictions

                st.markdown("### 🧾 Predictions:")
                st.dataframe(df[["text", "prediction"]].head(20))

                # Download button
                csv_download = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv_download,
                    file_name="fake_news_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")



#streamlit run app.py 
