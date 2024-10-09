import streamlit as st
import os
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Assuming you've set up your Google API key as before
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def clone_repository(repo_url, local_path):
    if not os.path.exists(local_path):
        Repo.clone_from(repo_url, local_path)
    return local_path

def load_code_files(repo_path):
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.py', '.js', '.java', '.cpp', '.html', '.css')):  # Add more extensions as needed
                try:
                    loader = TextLoader(os.path.join(root, file), encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
    return documents

def chunk_code(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = """
    You are an AI assistant specialized in analyzing and explaining code. Use the following pieces of code context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Provide code snippets when relevant.

    Code Context:
    {context}

    Question: {question}
    AI Assistant: Let's analyze this code:"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

def main():
    st.set_page_config("GitHub Code Analysis", layout="wide", page_icon=":computer:")
    st.title("GitHub Code Analysis with RAG")

    repo_url = st.text_input("Enter GitHub Repository URL:")
    if st.button("Analyze Repository"):
        with st.spinner("Cloning and analyzing repository..."):
            local_path = clone_repository(repo_url, "./temp_repo")
            documents = load_code_files(local_path)
            chunks = chunk_code(documents)
            vector_store = create_vector_store(chunks)
            st.session_state.chain = get_conversational_chain(vector_store)
            st.success("Repository analyzed successfully!")

    st.subheader("Ask about the Code")
    user_question = st.text_input("Your question about the code:")
    if user_question and 'chain' in st.session_state:
        with st.spinner("Analyzing..."):
            response = st.session_state.chain.run(user_question)
            st.markdown(f"**AI Assistant:** {response}")

    if st.button("Clear Conversation"):
        st.session_state.chain.memory.clear()
        st.success("Conversation cleared!")

if __name__ == "__main__":
    main()
