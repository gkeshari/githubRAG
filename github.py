# # import streamlit as st
# # import os
# # from git import Repo
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.document_loaders import TextLoader
# # from langchain.vectorstores import FAISS
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain.memory import ConversationBufferMemory
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.prompts import PromptTemplate

# # # Assuming you've set up your Google API key as before
# # import google.generativeai as genai
# # from dotenv import load_dotenv
# # load_dotenv()
# # os.getenv("GOOGLE_API_KEY")
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # def clone_repository(repo_url, local_path):
# #     if not os.path.exists(local_path):
# #         Repo.clone_from(repo_url, local_path)
# #     return local_path

# # def load_code_files(repo_path):
# #     documents = []
# #     for root, _, files in os.walk(repo_path):
# #         for file in files:
# #             if file.endswith(('.py', '.js', '.java', '.cpp', '.html', '.css')):  # Add more extensions as needed
# #                 try:
# #                     loader = TextLoader(os.path.join(root, file), encoding='utf-8')
# #                     documents.extend(loader.load())
# #                 except Exception as e:
# #                     print(f"Error loading {file}: {str(e)}")
# #     return documents

# # def chunk_code(documents):
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=1000,
# #         chunk_overlap=200,
# #         length_function=len,
# #         separators=["\n\n", "\n", " ", ""]
# #     )
# #     chunks = text_splitter.split_documents(documents)
# #     return chunks

# # def create_vector_store(chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
# #     return vector_store

# # def get_conversational_chain(vector_store):
# #     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# #     prompt_template = """
# #     You are an AI assistant specialized in analyzing and explaining code. Use the following pieces of code context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Provide code snippets when relevant.

# #     Code Context:
# #     {context}

# #     Question: {question}
# #     AI Assistant: Let's analyze this code:"""

# #     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
# #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
# #     chain = ConversationalRetrievalChain.from_llm(
# #         llm=model,
# #         retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
# #         memory=memory,
# #         combine_docs_chain_kwargs={"prompt": prompt}
# #     )

# #     return chain

# # def main():
# #     st.set_page_config("GitHub Code Analysis", layout="wide", page_icon=":computer:")
# #     st.title("GitHub Code Analysis with RAG")

# #     repo_url = st.text_input("Enter GitHub Repository URL:")
# #     if st.button("Analyze Repository"):
# #         with st.spinner("Cloning and analyzing repository..."):
# #             local_path = clone_repository(repo_url, "./temp_repo")
# #             documents = load_code_files(local_path)
# #             chunks = chunk_code(documents)
# #             vector_store = create_vector_store(chunks)
# #             st.session_state.chain = get_conversational_chain(vector_store)
# #             st.success("Repository analyzed successfully!")

# #     st.subheader("Ask about the Code")
# #     user_question = st.text_input("Your question about the code:")
# #     if user_question and 'chain' in st.session_state:
# #         with st.spinner("Analyzing..."):
# #             response = st.session_state.chain.run(user_question)
# #             st.markdown(f"**AI Assistant:** {response}")

# #     if st.button("Clear Conversation"):
# #         st.session_state.chain.memory.clear()
# #         st.success("Conversation cleared!")

# # if __name__ == "__main__":
# #     main()



# import streamlit as st
# import os
# import requests
# from git import Repo
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import TextLoader
# from langchain.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate

# # Assuming you've set up your Google API key as before
# import google.generativeai as genai
# from dotenv import load_dotenv
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # You'll need to set up a GitHub token for API access
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# def get_user_repositories(username):
#     headers = {"Authorization": f"token {GITHUB_TOKEN}"}
#     response = requests.get(f"https://api.github.com/users/{username}/repos", headers=headers)
#     if response.status_code == 200:
#         return [(repo['name'], repo['html_url']) for repo in response.json()]
#     else:
#         st.error(f"Error fetching repositories: {response.status_code}")
#         return []

# def clone_repository(repo_url, local_path):
#     if not os.path.exists(local_path):
#         Repo.clone_from(repo_url, local_path)
#     return local_path

# def load_code_files(repo_path):
#     documents = []
#     for root, _, files in os.walk(repo_path):
#         for file in files:
#             if file.endswith(('.py', '.js', '.java', '.cpp', '.html', '.css')):  # Add more extensions as needed
#                 try:
#                     loader = TextLoader(os.path.join(root, file), encoding='utf-8')
#                     documents.extend(loader.load())
#                 except Exception as e:
#                     print(f"Error loading {file}: {str(e)}")
#     return documents

# def chunk_code(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = text_splitter.split_documents(documents)
#     return chunks

# def create_vector_store(chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
#     return vector_store

# def get_conversational_chain(vector_store):
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     prompt_template = """
#     You are an AI assistant specialized in analyzing and explaining code. Use the following pieces of code context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Provide code snippets when relevant.

#     Code Context:
#     {context}

#     Question: {question}
#     AI Assistant: Let's analyze this code:"""

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=model,
#         retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": prompt}
#     )

#     return chain

# def main():
#     st.set_page_config("GitHub Code Analysis", layout="wide", page_icon=":computer:")
#     st.title("GitHub Code Analysis with RAG")

#     github_username = st.text_input("Enter GitHub Username:")
#     if github_username:
#         repos = get_user_repositories(github_username)
#         if repos:
#             selected_repo = st.selectbox("Select a repository", options=repos, format_func=lambda x: x[0])
#             if st.button("Analyze Repository"):
#                 with st.spinner("Cloning and analyzing repository..."):
#                     local_path = clone_repository(selected_repo[1], f"./temp_repo_{selected_repo[0]}")
#                     documents = load_code_files(local_path)
#                     chunks = chunk_code(documents)
#                     vector_store = create_vector_store(chunks)
#                     st.session_state.chain = get_conversational_chain(vector_store)
#                     st.success("Repository analyzed successfully!")
#         else:
#             st.warning("No repositories found or unable to fetch repositories.")

#     st.subheader("Ask about the Code")
#     user_question = st.text_input("Your question about the code:")
#     if user_question and 'chain' in st.session_state:
#         with st.spinner("Analyzing..."):
#             response = st.session_state.chain.run(user_question)
#             st.markdown(f"**AI Assistant:** {response}")

#     if st.button("Clear Conversation"):
#         if 'chain' in st.session_state:
#             st.session_state.chain.memory.clear()
#             st.success("Conversation cleared!")
#         else:
#             st.warning("No active conversation to clear.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import os
import requests
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import concurrent.futures
import tempfile
import pickle

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_data(ttl=3600)
def get_user_repositories(username):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(f"https://api.github.com/users/{username}/repos", headers=headers)
    if response.status_code == 200:
        return [(repo['name'], repo['html_url']) for repo in response.json()]
    else:
        st.error(f"Error fetching repositories: {response.status_code}")
        return []

def clone_repository(repo_url):
    with tempfile.TemporaryDirectory() as tmpdirname:
        Repo.clone_from(repo_url, tmpdirname)
        return tmpdirname

def load_code_file(file_path):
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def load_code_files(repo_path):
    documents = []
    code_extensions = {'.py', '.js', '.java', '.cpp', '.html', '.css'}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(load_code_file, file_path))
        
        for future in concurrent.futures.as_completed(futures):
            documents.extend(future.result())
    
    return documents

def create_or_load_vector_store(chunks, repo_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store_path = f"vector_store_{repo_name}.pkl"
    
    if os.path.exists(vector_store_path):
        with open(vector_store_path, "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        with open(vector_store_path, "wb") as f:
            pickle.dump(vector_store, f)
    
    return vector_store

def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = """
    You are an AI assistant specialized in analyzing and explaining code. Use the provided code snippets and context below to answer the question at the end.

    Ensure your response is well-structured and includes proper indentation for any code snippets.
    If you are unsure about the answer, clearly state that you do not know, rather than attempting to fabricate a response.
    Include relevant code snippets to support your explanations whenever applicable.

    Code Context:
    {context}

    Question: {question}
    AI Assistant: Let's analyze this code:"""

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, convert_system_message_to_human=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

def main():
    st.set_page_config("GitHub Code Analysis", layout="wide", page_icon=":computer:")
    st.title("GitHub Code Analysis with RAG")

    github_username = st.text_input("Enter GitHub Username:")
    if github_username:
        repos = get_user_repositories(github_username)
        if repos:
            selected_repo = st.selectbox("Select a repository", options=repos, format_func=lambda x: x[0])
            if st.button("Analyze Repository"):
                with st.spinner("Cloning and analyzing repository..."):
                    local_path = clone_repository(selected_repo[1])
                    documents = load_code_files(local_path)
                    chunks = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    ).split_documents(documents)
                    vector_store = create_or_load_vector_store(chunks, selected_repo[0])
                    st.session_state.chain = get_conversational_chain(vector_store)
                    st.success("Repository analyzed successfully!")
        else:
            st.warning("No repositories found or unable to fetch repositories.")

    st.subheader("Ask about the Code")
    user_question = st.text_input("Your question about the code:")
    if user_question and 'chain' in st.session_state:
        with st.spinner("Analyzing..."):
            response = st.session_state.chain.run(user_question)
            st.markdown(f"**AI Assistant:** {response}")

    if st.button("Clear Conversation"):
        if 'chain' in st.session_state:
            st.session_state.chain.memory.clear()
            st.success("Conversation cleared!")
        else:
            st.warning("No active conversation to clear.")

if __name__ == "__main__":
    main()