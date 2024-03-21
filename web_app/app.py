import streamlit as st
from utility import getSentiment



st.title("Sentiment Predictor using Word Embeddings")

# Text Input
corpus = st.text_area("Enter text here for Sentiment Analysis")
if corpus:

    # Prediction of Sentiment
    sentiment = getSentiment(corpus) 

    # Display Sentiment
    st.header('Sentiment')
    if sentiment == 1 and corpus!='':
        st.write("Positive")
    else:
        st.write("Negative")

