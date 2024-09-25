import dspy
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

INDEX_FOLDER = "konproz_index"

class FAISSRetriever(dspy.Retrieve):
    def __init__(self):
        self.vectorstore = self.load_faiss_index()

    def load_faiss_index(self):
        embeddings = HuggingFaceEmbeddings()
        return FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

    def __call__(self, query: str, k: int = 3) -> List[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

class Retriever(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = FAISSRetriever()

    def forward(self, query: str) -> List[str]:
        return self.retrieve(query, k=3)

def load_retriever():
    return Retriever()