import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from agent import load_agent

st.set_page_config(page_title="Doc Q&A", page_icon="🔍", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { padding-top: 2rem; }
    .header {
        text-align: center; padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #2a2a2a; margin-bottom: 2rem;
    }
    .header h1 { font-size: 1.8rem; font-weight: 600; color: #f0f0f0; margin: 0; letter-spacing: -0.5px; }
    .header p { color: #888; font-size: 0.9rem; margin-top: 0.4rem; }
    .doc-badge {
        display: inline-block; background: #1a1a2e; border: 1px solid #2d2d4e;
        color: #7c7cff; padding: 0.3rem 0.8rem; border-radius: 20px;
        font-size: 0.8rem; margin-bottom: 1.5rem;
    }
    .answer-box {
        background: #111; border: 1px solid #2a2a2a; border-left: 3px solid #7c7cff;
        border-radius: 8px; padding: 1.2rem 1.5rem; margin-top: 1rem;
        color: #e0e0e0; line-height: 1.7; font-size: 0.95rem;
    }
    .question-label {
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px; color: #555; margin-bottom: 0.5rem;
    }
    .answer-label {
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px; color: #7c7cff; margin-bottom: 0.5rem;
    }
    .stTextInput > div > div > input {
        background: #111 !important; border: 1px solid #2a2a2a !important;
        border-radius: 8px !important; color: #f0f0f0 !important; padding: 0.7rem 1rem !important;
    }
    .stButton > button {
        background: #7c7cff !important; color: white !important; border: none !important;
        border-radius: 8px !important; padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important; width: 100% !important;
    }
    .stButton > button:hover { background: #6060e0 !important; }
    .chunk-box {
        background: #0d0d0d; border: 1px solid #1e1e1e; border-radius: 6px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem; font-size: 0.82rem;
        color: #666; line-height: 1.5;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🔍 Document Q&A</h1>
    <p>Upload a PDF — ask anything about it</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

if uploaded_file is not None:
    if "ingested_file" not in st.session_state or st.session_state.ingested_file != uploaded_file.name:
        with st.spinner("Indexing document..."):
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings()
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            st.session_state.ingested_file = uploaded_file.name
            st.session_state.chain = load_agent()
            st.session_state.chat_history = []

    chain = st.session_state.chain
    st.markdown(f'<div class="doc-badge">📄 {uploaded_file.name}</div>', unsafe_allow_html=True)

    st.markdown('<div class="question-label">Your Question</div>', unsafe_allow_html=True)
    question = st.text_input("", placeholder="What is this document about?", label_visibility="collapsed")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            result = chain.invoke({
                "question": question,
                "chat_history": st.session_state.chat_history
            })
            answer = result["answer"]
            st.session_state.chat_history.append((question, answer))

        st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        with st.expander("View source chunks"):
            vectorstore = chain.retriever.vectorstore
            results = vectorstore.similarity_search_with_relevance_scores(question, k=6)
            seen = set()
            unique_results = []
            for doc, score in results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_results.append((doc, score))
                if len(unique_results) == 3:
                    break
            for i, (doc, score) in enumerate(unique_results):
                pct = round(score * 100, 1)
                bar_color = "#4caf50" if pct >= 70 else "#ff9800" if pct >= 40 else "#f44336"
                page = doc.metadata.get("page", "?")
                st.markdown(f'''
                <div class="chunk-box">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                        <strong>Chunk {i+1}</strong> — page {page}
                        <span style="color:{bar_color}; font-weight:600;">{pct}% match</span>
                    </div>
                    {doc.page_content}
                </div>''', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #444;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">📂</div>
        <div style="font-size: 0.9rem;">Drop a PDF above to get started</div>
    </div>
    """, unsafe_allow_html=True)