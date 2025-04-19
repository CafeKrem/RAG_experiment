from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import gradio as gr

# === Load Documents ===
documents = SimpleDirectoryReader("documents").load_data()

# === Setup Embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Setup ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# === Setup Quantized LLM from local GGUF file
llm = LlamaCPP(
    model_path="./models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",  # update path if needed
    temperature=0.7,
    max_new_tokens=256,
    context_window=1024,
    model_kwargs={
        "n_ctx": 1024,
        "n_threads": 4
    }
)

# ✅ Register models globally with new Settings API
Settings.llm = llm
Settings.embed_model = embed_model

# === Build the Vector Index (no ServiceContext needed)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# === Query Engine
query_engine = index.as_query_engine()

# === Gradio UI
def ask_rag(query):
    response = query_engine.query(query)
    return str(response)

gr.Interface(
    fn=ask_rag,
    inputs="text",
    outputs="text",
    title="🧠 Local RAG Chat (TinyLlama)",
    description="Runs fully offline using ChromaDB + TinyLlama on CPU"
).launch()
