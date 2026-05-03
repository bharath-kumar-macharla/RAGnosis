import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

load_dotenv()

DB_DIR = "./chroma_langchain_db"
PDF_PATH = "attention-is-all-you-need.pdf"

# -----------------------------
# Load & Split Documents
# -----------------------------

def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)

# -----------------------------
# Create / Load Vector Store
# -----------------------------

def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return Chroma(
        collection_name="research_collection",
        embedding_function=embedding_model,
        persist_directory=DB_DIR
    )

# -----------------------------
# Add Docs Only If Empty
# -----------------------------

def index_documents(vector_store):
    if len(vector_store.get()["ids"]) == 0:
        print("📄 Indexing documents...")
        splits = load_and_split(PDF_PATH)
        vector_store.add_documents(splits)
        print("✅ Indexing complete")
    else:
        print("⚡ Using existing vector DB")

# -----------------------------
# Retrieval
# -----------------------------

def retrieve_context(vector_store, query, k=2):
    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in docs
    )

    return context, docs

# -----------------------------
# LLM Setup
# -----------------------------

def get_model():
    return init_chat_model(
        "google_genai:gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
# -----------------------------
# Initializing the functions
# -----------------------------

vector_store = get_vector_store()
index_documents(vector_store)
model = get_model()

# -----------------------------
# RAG Pipeline
# -----------------------------

def rag_pipeline(query):
    context, docs = retrieve_context(vector_store, query)

    system_prompt = f"""
    You are a helpful assistant.
    Use ONLY the context below to answer.
    Do not hallucinate.

    Context:
    {context}
    """

    response = model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])

    return {
        "answer": response.content,
        "sources": docs
    }

# -----------------------------
# Chat function for communitation
# -----------------------------

def chat(query):
    try:
        result = rag_pipeline(query)
        return result["answer"]
    except Exception as e:
        return f"Error: {str(e)}"
    
# -----------------------------
# UI using Gradio
# -----------------------------

app = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        placeholder="Ask something about your document...",
        label="Your Question"
    ),
    outputs=gr.Textbox(label="Answer"),
    title="🧠 RAGnosis",
    description="Chat with your local document using AI"
)

if __name__ == "__main__":
    app.launch()

