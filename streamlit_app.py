import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="[Machine-Learning for NLP]", page_icon="👾", layout='centered', initial_sidebar_state='auto', menu_items=None)

st.header("Machine Learning")
st.subheader('\t NLP Predictions/Multinomial Nayves bayes')
st.subheader('')
with open('model.mdl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

vectorizer = pickle.load(open("vector.pickel", "rb"))

st.write("Please write a review and our machine learning model will analyse it 😊")
#st.markdown("![Alt Text](https://media.giphy.com/media/kNwQN4ueScpbaeWtef/giphy.gif)")
inputStr = st.text_input("Review ")

vectorizer = vectorizer[0]
model = model[0]

if  st.button('Make prediction') and len(inputStr) > 0:
    vector = vectorizer.transform([inputStr])
    y_pred = model.predict(vector.toarray()).ravel()
    res = y_pred[0]
    print(res)
    if res == 0:
        st.write("The review is negative")
        st.markdown("![Alt Text](https://media.giphy.com/media/8OVBzeDTFe8XQfivmP/giphy.gif)")
    if res == 1:
        st.write("The review is positive")
        st.markdown("![Alt Text](https://media.giphy.com/media/l0MYKDrj6SXHz8YYU/giphy.gif)")
