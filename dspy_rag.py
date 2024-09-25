import dspy
from dspy_retriever import load_retriever
from dspy_generator import load_generator

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retriever = load_retriever()
        self.generator = load_generator()

    def forward(self, query: str) -> str:
        retrieved_docs = self.retriever(query)
        context = "\n".join(retrieved_docs)
        return self.generator(query=query, context=context)

def load_rag_pipeline():
    return RAG()