from pathlib import Path
import streamlit as st
import joblib

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")

st.title("SMS Spam Classifier")

st.write(
    "This is a simple machine learning model that can take any SMS and determine whether it is SPAM or a legitimate message(HAM)"
)

st.write(
    "You can copy or enter any message you have received from your SMS on phone and paste it here"
)

message = st.text_area("Enter SMS Message")

if st.button("Check Message"):
    message = message.lower()

    vector_mess = vectorizer.transform([message])

    model_mess = model.predict_proba(vector_mess)[0][1]

    if model_mess >= 0.5:
        st.error("This message is SPAM")
    else:
        st.success("This message is legitimate (HAM)")

st.caption("Built using Balanced Logistic Regression and Bag of Words")
