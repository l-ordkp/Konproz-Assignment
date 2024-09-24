
from query import load_faiss_index, retrieve
from aug import get_gemini_response

def process_query(index_folder, query):
    vectorstore = load_faiss_index(index_folder)
    retrieved_docs = retrieve(vectorstore, query)
    context = "\n".join(retrieved_docs)
    answer = get_gemini_response(query, context)
    return answer

if __name__ == "__main__":
    index_folder = "konproz_index"
    
    # Example usage
    query = "What is the main topic of the document?"
    answer = process_query(index_folder, query)
    print(f"Question: {query}")
    print(f"Answer: {answer['parts'][0]['text']}")