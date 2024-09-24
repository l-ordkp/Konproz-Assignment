# retrieval.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_faiss_index(index_folder):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

def retrieve(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    index_folder = "konproz_index"
    vectorstore = load_faiss_index(index_folder)
    
