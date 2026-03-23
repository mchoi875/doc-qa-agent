cat > README.md << 'EOF'
# AI Document Q&A Agent

A RAG-based AI agent that answers questions about any PDF document using only the document's content.

## How it works
1. Upload a PDF through the browser interface
2. The document is chunked and embedded into a ChromaDB vector store
3. Questions are matched to the most relevant chunks via semantic search
4. OpenAI GPT-3.5 generates a grounded answer with prompt guardrails to prevent hallucination

## Tech Stack
- Python, LangChain, OpenAI API, ChromaDB, Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
EOF
