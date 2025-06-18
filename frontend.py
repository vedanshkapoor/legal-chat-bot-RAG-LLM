#step1: setup upload PDF functionality
import streamlit as st
from rag_pipeline import answer_query,retrieve_docs, llm_model

uploaded_file = st.file_uploader("Upload your PDF here",
                                 type = "pdf",
                                 accept_multiple_files=False)
#step2: chatbot skeleton (Question and answer)
user_query = st.text_area("Enter your prompt", height = 150, placeholder = "Ask anything!")

ask_question = st.button("Ask your question to AI Lawyer")
if ask_question:
    #rag pipeline
    if uploaded_file:
        st.chat_message("user").write(user_query)

        # rag pipeline
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        #fixed_response = "Hi, this is a fixed response"
        st.chat_message("AI Lawyer").write(response)
    else:
        st.error("kindly upload a valid PDF file first!")



#