# 📄 Genius PDF Chatbot

# 📄 PDF Chatbot™ by Rupesh Ghimire

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pdf-genius-chatbot.streamlit.app/)

**[Click here to Try the Live Demo](https://pdf-genius-chatbot.streamlit.app/)**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
...
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)

A Python-based RAG (Retrieval-Augmented Generation) application that allows you to upload PDF documents and chat with them using Google's Gemini AI. 

Built with **Streamlit**, **LangChain**, and **Google Generative AI**.

## 🚀 Features
* **Interactive Chat UI:** Modern chat interface with message history.
* **PDF Analysis:** Upload any PDF to extract text automatically.
* **Vector Search:** Uses FAISS and Google Embeddings to find the exact paragraphs relevant to your question.
* **Gemini 2.0 Integration:** Powered by Google's latest `gemini-2.0-flash-001` model for fast and accurate responses.

## 🛠️ Prerequisites
* Python 3.9 or higher.
* A Google Cloud API Key (from [Google AI Studio](https://aistudio.google.com/)).

## 📦 Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repo-url>
    cd my_pdf_genius
    ```

2.  **Create and Activate a Virtual Environment:**
    * *macOS/Linux:*
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * *Windows:*
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🔑 Configuration

1.  Create a file named `.env` in the root directory of the project.
2.  Open the file and add your Google API Key:
    ```ini
    GOOGLE_API_KEY="AIzaSy...<your-api-key-here>"
    ```

## 🏃‍♂️ Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run chatbot.py
    ```

2.  The app will open in your browser (usually at `http://localhost:8501`).
3.  **Upload a PDF** using the sidebar on the left.
4.  Wait for the "File Processed Successfully!" message.
5.  **Start chatting** in the main window!


⚠️ Troubleshooting
Error: 400 API Key Expired: * Check your .env file.

If you just created the key, wait 2-5 minutes for it to activate.

Error: Model not found: * Ensure you are using langchain-google-genai version 1.0 or higher.

Check chatbot.py to ensure the model name matches what is available in your region (e.g., gemini-pro or gemini-1.5-flash).


## 📂 Project Structure

```text
my_first_chatbot/
├── .env                  # API Key (DO NOT COMMIT THIS TO GITHUB)
├── chatbot.py            # Main application code
├── requirements.txt      # List of python libraries
├── README.md             # Project documentation
└── venv/                 # Virtual environment folder



