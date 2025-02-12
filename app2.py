import os
import io
import pandas as pd
import numpy as np
import faiss
import traceback
import streamlit as st
from sentence_transformers import SentenceTransformer
import openai

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain.schema import Document

# ------------------------------
# SETUP & GLOBAL VARIABLES
# ------------------------------

# Set page configuration
st.set_page_config(page_title="Excel Chat Bot", layout="wide")

# Hide Streamlit style (optional)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Retrieve OpenAI API key (use Streamlit secrets or environment variable)
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Create OpenAI client
client = openai.OpenAI(api_key=api_key)

# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

def parse_question_with_llm(question, chat_history):
    """
    Refine the user's question using OpenAI's GPT model, maintaining context.
    """
    try:
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based strictly on the provided Excel data."}
        ]
        # Include last 5 exchanges in the chat history
        for user_input, bot_response in chat_history[-5:]:
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": bot_response})
        messages.append({"role": "user", "content": question})
    
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # update to your desired model
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error using GPT model: {str(e)}"

def load_and_preprocess_data(uploaded_file):
    """
    Load and preprocess the uploaded Excel file.
    """
    try:
        # Read file into a pandas DataFrame
        df = pd.read_excel(uploaded_file)
        df.fillna("", inplace=True)
        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        # Create a combined column for embedding (you can adjust the formatting)
        df["combined"] = df.apply(lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
        st.session_state.df = df
        return f"Data loaded successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    except Exception as e:
        return f"Error loading file: {str(e)}"

def setup_vectorstore(index, df):
    """
    Set up a FAISS vectorstore using langchain's InMemoryDocstore.
    """
    docs = [
        Document(page_content=row["combined"], metadata={col: row[col] for col in df.columns})
        for _, row in df.iterrows()
    ]
    docstore = InMemoryDocstore({str(idx): doc for idx, doc in enumerate(docs)})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    def embedding_function(texts):
        return st.session_state.embedding_model.encode(texts, convert_to_tensor=False)
    
    return LC_FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_function
    )

def initialize_faiss():
    """
    Initialize the FAISS index with embeddings from the DataFrame.
    """
    df = st.session_state.df
    if df is None:
        return "Please upload an Excel file first."

    try:
        texts = df["combined"].tolist()
        embeddings = st.session_state.embedding_model.encode(texts, convert_to_tensor=True, batch_size=16)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings.cpu()))
        st.session_state.faiss_index = index
        st.session_state.vectorstore = setup_vectorstore(index, df)
        return "FAISS index initialized successfully."
    except Exception as e:
        return f"Error initializing FAISS: {str(e)}\n{traceback.format_exc()}"

def ask_question(question):
    """
    Process the user's question, perform a similarity search on the FAISS index,
    and generate an answer using the GPT model.
    """
    if st.session_state.vectorstore is None:
        return "Please initialize FAISS before asking questions."

    try:
        # Refine the question using GPT (with context)
        refined_question = parse_question_with_llm(question, st.session_state.chat_history)
        
        # Encode and search for relevant data
        query_embedding = st.session_state.embedding_model.encode([refined_question], convert_to_tensor=True).cpu().numpy()
        distances, indices = st.session_state.faiss_index.search(query_embedding, k=5)
        
        # Gather relevant rows
        df = st.session_state.df
        results = []
        for idx in indices[0]:
            if idx < len(df):
                row = df.iloc[idx]
                results.append(row.to_dict())
        
        if not results:
            return "No matching results found."

        # Create a context string from results
        context = "\n".join([
            " | ".join([f"{k}: {v}" for k, v in result.items()])
            for result in results
        ])
        prompt = (
            "Extract relevant information from the following data and answer strictly based on it. "
            "Do not generate unrelated information.\n\n"
            f"Data:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # update to your desired model
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        return f"Error answering question: {str(e)}\n{traceback.format_exc()}"

# ------------------------------
# STREAMLIT UI LAYOUT
# ------------------------------

st.title("📊 Befach Data Chat Bot")
st.markdown("""
Welcome to the **Befach Data Chat Bot**. Upload your Excel file, initialize the FAISS index, 
and ask questions to get insights based strictly on your data.
""")

# Create two columns for a cleaner layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        load_status = load_and_preprocess_data(uploaded_file)
        st.success(load_status)
    else:
        st.info("Please upload an Excel (.xlsx) file.")

    st.header("2. Initialize FAISS")
    if st.button("Initialize FAISS"):
        init_status = initialize_faiss()
        if "successfully" in init_status.lower():
            st.success(init_status)
        else:
            st.error(init_status)

with col2:
    st.header("3. Chat with Your Data")
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input("Enter your question", placeholder="Type your question here...")
        submit_button = st.form_submit_button("Submit")
    
    if submit_button and user_question:
        with st.spinner("Processing your question..."):
            answer = ask_question(user_question)
            # Append the new exchange to the chat history
            st.session_state.chat_history.append((user_question, answer))

    st.subheader("Chat History")
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
    else:
        st.info("Your conversation will appear here.")
