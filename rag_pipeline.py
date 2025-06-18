from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
#step1 setup LLM
llm_model = ChatGroq(model = "deepseek-r1-distill-llama-70b")

#step2 Retrieve docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


#step3 answers questions
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you dont know the answer , just say that you dont know the answer. dont try to make up an answer.
Dont provide anything out of the given context
Question: {question}
Context: {context}
Answer: 
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model  #piping function
    return chain.invoke({"question":query, "context":context})


#question = "according to recent years which article has been most violated and why ?"
#retrieved_docs = retrieve_docs(question)
#print("AI lawyer", answer_query(documents = retrieved_docs, model = llm_model, query = question))


