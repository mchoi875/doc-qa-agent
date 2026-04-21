# AI Document Q&A Agent

Built to explore production RAG patterns including hallucination guardrails and retrieval transparency

A RAG-based AI agent that answers questions about any PDF document using only the document's content.

## Features
- RAG Pipeline — ingests PDFs, chunks and embeds content into a ChromaDB vector store, retrieves semantically relevant chunks to answer questions
- Conversation Memory — maintains context across questions within a session, enabling follow-up questions without repeating context
- Relevance Scoring — displays color-coded match percentages on retrieved source chunks for transparency and auditability
- Prompt Guardrails — answers are grounded strictly to the uploaded document; responds with "I don't have enough information" when the answer isn't in the document
- Streamlit UI — browser-based interface for real-time PDF upload and conversational Q&A

## How it works
1. Upload a PDF through the browser interface
2. The document is chunked and embedded into a ChromaDB vector store
3. User asks a question and it's embedded and matched against stored chunks via semantic search
4. Top matching chunks + conversation history are sent to GPT-3.5 to generate a grounded answer
5. Source chunks with relevance scores are displayed for auditability

## Tech Stack
- Python, LangChain, OpenAI API, ChromaDB, Streamlit

## Run Locally
```bash
# Install dependencies
pip install langchain langchain-openai langchain-community langchain-text-splitters chromadb pypdf streamlit openai

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run the app
streamlit run app.py
```
