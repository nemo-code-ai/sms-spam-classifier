from pathlib import Path
import streamlit as st
import joblib

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")

st.title("SMS Spam Classifier")

message = st.text_area("Enter SMS Message")

if st.button("Text Message"):
    message = message.lower()

    vector_mess = vectorizer.transform([message])

    model_mess = model.predict_proba(vector_mess)[0][1]

    if model_mess >= 0.5:
        st.success("This message is SPAM")
    else:
        st.info("This message is HAM")
