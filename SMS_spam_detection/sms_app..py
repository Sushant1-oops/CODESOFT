import streamlit as st
import joblib
model=joblib.load("sms_model.pkl")
vectorizer=joblib.load("vectorizer.pkl")
st.set_page_config("SMS SPAM DETECTION")
st.title("SMS SPAM DETECTION")
st.subheader("Enter an SMS message below and find out if it's Spam or Ham.")

input=st.text_area("Type your message here")
if st.button("check message"):
    if input.strip()=="":
        st.warning("Please enter the messgae")git push -u origin master

    else:
        input_vector=vectorizer.transform([input])
        prediction=model.predict(input_vector)[0]
        if prediction==1:
            st.error("Message is Spam")
        else:
            st.success("Message is Ham")

