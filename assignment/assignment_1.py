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
trained_w2v = gensim.models.Word2Vec.load("data/w2v-advanced.model")

# App title
st.set_page_config(page_title="Word2vec Question and Answer Chatbot")

# Add header image 
st.image("data/header-chat-box.png")

# chat title 
st.title("Word2vec Question and Answer Chatbot")

#st.session_state.messages = [] # reset session state

# Store generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
#else:
  #  st.session_state.pop('messages') # reset session state
   # st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
	if type(message["content"]) == str:
		with st.chat_message(message["role"]):
			st.write(message["content"])
	else:
		with st.chat_message(message["role"]):
			st.image(message["content"])


# Function for generating response for query question
def trained_sentence_vec(sent):
    # Filter out terms that are not in the vocabulary from the question sentence
    qu_voc = [tm for tm in sent if tm in trained_w2v.wv]
    # Get the embedding of the characters
    emb = np.vstack([trained_w2v.wv[tm] for tm in qu_voc])
    # Calculate the vectors of each included word to get the vector of the question
    ave_vec = np.mean(emb, axis=0)
    return ave_vec

def find_answer(qr_sentence, ques_vec, ans_vec):
    # use one query sentence to retrieve answer
    qr_sentence = gensim.utils.simple_preprocess(qr_sentence)
    qr_sent_vec = trained_sentence_vec(qr_sentence)

    # perform vector search through similarity comparison
    n_dim = ques_vec.shape[1]
    n_q_a = ques_vec.shape[0] # number of pairs of question and answer
    x = np.vstack(ques_vec).astype(np.float32)
    y = np.vstack(ans_vec).astype(np.float32)
    q = qr_sent_vec.reshape(1, -1)
    index = faiss.index_factory(n_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    # add all questions
    faiss.normalize_L2(x)
    index.add(x)
    # add all answers
    faiss.normalize_L2(y)
    index.add(y)
    # do vector search for the query sentence
    faiss.normalize_L2(q)
    similarity, idx = index.search(q, k=index.ntotal)
    ans_idx = idx[0][0]
    if ans_idx >= n_q_a:
      ans_idx = ans_idx - n_q_a
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
            ans_idx = find_answer(prompt, ques_vec, ans_vec)
            response = df["Answer"][ans_idx]
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
