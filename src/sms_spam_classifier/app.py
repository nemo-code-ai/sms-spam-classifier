import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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
