"""
Gemini PDF Chatbot
Copyright (c) 2025 Rupesh Ghimire

This software is the intellectual property of Rupesh Ghimire.
"""

import streamlit as st
import os
from dotenv import load_dotenv

# LangChain & AI Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Chatbot™", page_icon="📄", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
def setup_sidebar_ui():
    st.markdown("""
    <style>
        /* 1. Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            border-right: 1px solid #e0e0e0;
        }

        /* 2. File Uploader Styling */
        [data-testid="stFileUploader"] {
            width: 100%;
        }
        [data-testid="stFileUploader"] section {
            background-color: #ffffff;
            border: 1px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #45a049;
            background-color: #f9fff9;
        }

        /* 3. Headers & Success Boxes */
        .sidebar-header {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }
        .success-box {
            padding: 15px;
            background-color: #d4edda;
            color: #155724;
            border-radius: 5px;
            margin-top: 10px;
            border: 1px solid #c3e6cb;
        }

        /* 4. Footer Styling (Now Static - No Overlap) */
        .sidebar-footer {
            width: 100%;
            text-align: center;
            font-size: 12px;
            color: #666;
            padding-top: 30px;
            margin-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)

def show_footer():
    st.markdown("""
    <div class="sidebar-footer">
        © 2025 <b>Rupesh Ghimire</b><br>
        All Rights Reserved ™<br>
        <br>
        <small>Powered by Gemini 2.0</small>
    </div>
    """, unsafe_allow_html=True)

setup_sidebar_ui()
st.title("📄 PDF Chatbot™")

# --- 2. SIDEBAR & FILE UPLOAD ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">📂 Document Hub</div>', unsafe_allow_html=True)

    # "Clear Chat" Button
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Modified: accept_multiple_files=True
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        # Create a unique ID for the set of files to detect changes
        current_file_names = "".join([f.name for f in uploaded_files])

        if "last_uploaded_set" not in st.session_state or st.session_state.last_uploaded_set != current_file_names:

            with st.spinner("🔄 Processing all PDFs..."):
                try:
                    all_text = ""
                    # Loop through all uploaded files
                    for pdf in uploaded_files:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            all_text += page.extract_text()

                    # Chunk text
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators="\n",
                        chunk_size=1000,
                        chunk_overlap=150,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(all_text)

                    # Embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004",
                        google_api_key=GOOGLE_API_KEY,
                        transport="rest",
                        task_type="retrieval_document"
                    )

                    # Vector Store
                    vector_store = FAISS.from_texts(chunks, embeddings)

                    # Save state
                    st.session_state.vector_store = vector_store
                    st.session_state.last_uploaded_set = current_file_names

                except Exception as e:
                    st.error(f"Error processing files: {e}")

        if "vector_store" in st.session_state:
            st.markdown(f"""
                <div class="success-box">
                    ✅ <b>{len(uploaded_files)} File(s) Processed</b><br>
                    Ready to chat!
                </div>
            """, unsafe_allow_html=True)

    else:
        st.info("👆 Upload PDFs to start analyzing.")

    show_footer()

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "vector_store" not in st.session_state:
        st.error("Please upload a PDF file first!")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                docs = st.session_state.vector_store.similarity_search(prompt)

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-001",
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY
                )

                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=prompt)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred: {e}")