from sentence_transformers import SentenceTransformer
import chromadb

# Initialize Chroma
client = chromadb.Client()
collection = client.get_or_create_collection("multimodal_rag")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_to_vector_store(text_chunks):
    ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    embeddings = embedder.encode(text_chunks).tolist()
    collection.add(documents=text_chunks, ids=ids, embeddings=embeddings)

def query_vector_store(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []
