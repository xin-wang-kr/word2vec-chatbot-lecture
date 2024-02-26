### Assignment instruction
# The goal of this assignment is to create a word2vec-based question-answer chatbot application that should be able
# to give the best answer based on vector search toward both question set and answer set. 
# Our exercise only showed how to apply question set for vector search. You can follow the hints to generate the chatbot. 
# What you need to submit for this assignment: an app url (You should publish your chatbot application on Streamlit Cloud. 
# Your chatbot assignment will be evaluated based on three query questions:
# (1) A year before improving and popularizing the electrophorus, what did Volta become?
# (2) Does the Hymenoptera order include ants?
# (3) Who invented the voltaic pile?
###

import streamlit as st
import pandas as pd
import faiss
import gensim
import numpy as np

# load question-answer dataset 
df = pd.read_csv("data/Question_Answer_Dataset_v1.2_S10.csv")

# load question and answer vectors

vector = np.load('data/vector.npz')
ques_vec = vector['x']
ans_vec = vector['y']

# load th trained word2vec model 
trained_w2v = gensim.models.Word2Vec.load("data/w2v.model")

# App title
st.set_page_config(page_title="Word2vec Question and Answer Chatbot")

# Add header image 
st.image("data/header-chat-box.png")

# chat title 
st.title("Word2vec Question and Answer Chatbot")

# Store generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messagess
for message in st.session_state.messages:
	if type(message["content"]) == str:
		with st.chat_message(message["role"]):
			st.write(message["content"])
	else:
		with st.chat_message(message["role"]):
			st.image(message["content"])


# Function to generate the embedding for query question
def trained_sentence_vec(sent):
    # Filter out terms that are not in the vocabulary from the question sentence
    # Hint: Use model.wv to get the whole vocabulary
    qu_voc = ...
    # Get the embedding of the characters
    # Hint: Stack arrays in sequence vertically using np.vstack
    emb = ...
    # Calculate the arithmetic mean for the vectors of each included word along the column 
    # to get the vector of the question
    ave_vec = ...
    return ave_vec

# Function to find the answer through vector search
def find_answer(qr_sentence, ques_vec):
    # use one query sentence to retrieve answer
    qr_sentence = gensim.utils.simple_preprocess(qr_sentence)
    qr_sent_vec = trained_sentence_vec(qr_sentence)

    # perform vector search through similarity comparison
    n_dim = ques_vec.shape[1]
    x = np.vstack(ques_vec).astype(np.float32)
    q = qr_sent_vec.reshape(1, -1)
    index = faiss.index_factory(n_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(x)
    index.add(x)
    faiss.normalize_L2(q)
    similarity, idx = index.search(q, k=index.ntotal)
    ans_idx = idx[0][0]
    return ans_idx


# User-provided prompt
if prompt := st.chat_input("What's your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans_idx = find_answer(prompt, ques_vec)
            response = df["Answer"][ans_idx]
            st.write(response)
            
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
