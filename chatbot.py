"""
Genius PDF Chatbot
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


def show_footer():
    st.markdown("""
    <style>
        /* 1. Target the sidebar to make space for the footer */
        [data-testid="stSidebar"] > div:first-child {
            padding-bottom: 100px; /* Prevent content from being hidden behind footer */
        }

        /* 2. Style the fixed footer */
        .sidebar-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 20rem; /* Matches default sidebar width */
            padding: 20px;
            background-color: #f0f2f6; /* Matches sidebar background */
            border-top: 1px solid #e0e0e0;
            text-align: center;
            font-size: 12px;
            color: #666;
            z-index: 99;
        }

        .sidebar-footer a {
            color: #666;
            text-decoration: none;
        }

        .sidebar-footer a:hover {
            color: #333;
            text-decoration: underline;
        }
    </style>

    <div class="sidebar-footer">
        © 2025 <b>Rupesh Ghimire</b><br>
        All Rights Reserved ™<br>
        <br>
        <small>Powered by Gemini 2.0</small>
    </div>
    """, unsafe_allow_html=True)

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="My PDF Genius", page_icon="📄", layout="wide")


# --- CUSTOM CSS FOR STYLING ---
def setup_sidebar_ui():
    st.markdown("""
    <style>
        /* 1. Style the Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            border-right: 1px solid #e0e0e0;
        }

        /* 2. Style the 'Browse Files' Button */
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

        /* 3. Custom Header Styling */
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
    </style>
    """,unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "© 2025 Rupesh Ghimire<br>All Rights Reserved",
        unsafe_allow_html=True)


# Apply the styles
setup_sidebar_ui()

st.title("📄 Chat with your PDF")

# --- 2. SIDEBAR & FILE UPLOAD ---
with st.sidebar:
    # Custom HTML Header
    st.markdown('<div class="sidebar-header">📂 Document Hub</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Use an expander to keep it clean, or just a container
    with st.container():
        st.write("**Upload your PDF below:**")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    # Status Indicator
    if uploaded_file is not None:
        # Check if we already processed this specific file
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:

            with st.spinner("🔄 Processing your PDF..."):
                try:
                    # A. Read PDF
                    pdf_reader = PdfReader(uploaded_file)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()

                    # B. Chunk text
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators="\n",
                        chunk_size=1000,
                        chunk_overlap=150,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(pdf_text)

                    # C. Generate Embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004",
                        google_api_key=GOOGLE_API_KEY,
                        transport="rest",
                        task_type="retrieval_document"
                    )

                    # D. Create Vector Store
                    vector_store = FAISS.from_texts(chunks, embeddings)

                    # E. Save to Session State
                    st.session_state.vector_store = vector_store
                    st.session_state.last_uploaded = uploaded_file.name

                except Exception as e:
                    st.error(f"Error processing file: {e}")

        # Show Success Message (Styled)
        if "vector_store" in st.session_state:
            st.markdown(f"""
                <div class="success-box">
                    ✅ <b>{uploaded_file.name}</b><br>
                    Ready to chat!
                </div>
            """, unsafe_allow_html=True)

            # Add a clear button to reset
            if st.button("🗑️ Remove File", use_container_width=True):
                del st.session_state["vector_store"]
                del st.session_state["last_uploaded"]
                st.rerun()

    else:
        # Show a helpful tip if no file is uploaded
        st.info("👆 Upload a PDF to start analyzing.")
show_footer()

# --- 3. CHAT INTERFACE ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question about your PDF..."):

    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Check if PDF is processed
    if "vector_store" not in st.session_state:
        st.error("Please upload a PDF file first!")
        st.stop()

    # 3. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant chunks from the vector store
                docs = st.session_state.vector_store.similarity_search(prompt)

                # Setup the Gemini Model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-001",
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY
                )

                # Create the chain
                chain = load_qa_chain(llm, chain_type="stuff")

                # Get the answer
                response = chain.run(input_documents=docs, question=prompt)

                st.markdown(response)

                # Save Assistant Message
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred: {e}")