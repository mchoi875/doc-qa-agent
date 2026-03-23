from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_agent():
    # 1. Load the existing ChromaDB vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # 2. Set up the retriever — grab the 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Custom prompt — this is the guardrail
    prompt_template = """
    You are a helpful assistant. Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't have enough information in this document to answer that."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 4. Wire everything together
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

if __name__ == "__main__":
    chain = load_agent()
    print("Agent ready. Type 'quit' to exit.\n")
    
    while True:
        question = input("Ask a question: ")
        if question.lower() == "quit":
            break
        result = chain.invoke({"query": question})
        print(f"\nAnswer: {result['result']}\n")