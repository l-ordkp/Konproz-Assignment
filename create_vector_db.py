

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return [doc.page_content for doc in texts]

def create_faiss_index(texts, index_name):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(index_name)
    print(f"FAISS index saved as {index_name}")

if __name__ == "__main__":
    pdf_path = "G S T Smart Guide.pdf"
    index_name = "konproz_index"
    
    texts = load_and_process_pdf(pdf_path)
    create_faiss_index(texts, index_name)