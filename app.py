import streamlit as st
import os
from ingest import ingest_pdf
from agent import load_agent

st.set_page_config(page_title="Doc Q&A Agent", page_icon="🤖")
st.title("📄 AI Document Q&A Agent")
st.caption("Upload a PDF and ask questions about it")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Ingest the PDF
    with st.spinner("Reading and indexing your document..."):
        ingest_pdf("temp.pdf")
    
    st.success(f"✅ Document ready: {uploaded_file.name}")
    
    # Load the agent
    chain = load_agent()
    
    # Chat interface
    st.divider()
    st.subheader("Ask a question about your document")
    
    question = st.text_input("Your question:", placeholder="What is this document about?")
    
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            result = chain.invoke({"query": question})
            answer = result["result"]
        
        st.markdown("### Answer")
        st.write(answer)
        
        # Show sources
        with st.expander("📚 View retrieved chunks"):
            embeddings_info = f"Retrieved top 3 most relevant chunks from your document to answer this question."
            st.caption(embeddings_info)

else:
    st.info("👆 Upload a PDF above to get started")