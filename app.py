import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the LSTM model
model = load_model('next_word_lstm.h5')

## Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


## Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_length-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

    ## Streamlit app
st.title("Next Word Prediction using LSTM")
input_text = st.text_input("Enter a sentence to predict the next word:", "to be or not to be")


if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1 ## Retreive the max sequence length from the model
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write(f"The predicted next word is: **{next_word}**")


