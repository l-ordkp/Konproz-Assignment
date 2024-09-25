import dspy
from dspy_rag import load_rag_pipeline
from aug import get_gemini_response, gemini_api_key



def process_query(query):
    # Set up the model and compile the RAG pipeline
    gemini = dspy.Google("models/gemini-1.0-pro",api_key=gemini_api_key, temperature=0.7)
    dspy.settings.configure(lm=gemini, max_tokens=1024)
    rag_pipeline = load_rag_pipeline()
    dspy_response = rag_pipeline(query)
    return get_gemini_response(dspy_response,query)

if __name__ == "__main__":
    # Example usage
    query = "What aspects does this book cover"
    answer = process_query(query)
    print(f"Question: {query}")
    print(f"Answer: {answer['parts'][0]['text']}")
