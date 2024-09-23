import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb

# Step 1: Extract Text from PDF (handling large PDFs)
pdf_file = "C:\\Users\\Kshit\\Desktop\\Konproz\\G S T Smart Guide.pdf"
doc = fitz.open(pdf_file)

# Initialize an empty list to hold all text chunks
text_chunks = []

# Extract text from each page and store in the list
for page_num in range(len(doc)):
    page = doc.load_page(page_num)  # Get each page
    page_text = page.get_text("text")  # Extract text from the page
    
    # Split page into smaller chunks (e.g., every 500 characters or by paragraph)
    chunks = [page_text[i:i + 500] for i in range(0, len(page_text), 500)]
    
    # Append the chunks to the text_chunks list
    text_chunks.extend(chunks)

doc.close()

# Check the number of chunks extracted
print(f"Total chunks extracted from the PDF: {len(text_chunks)}")
# Step 2: Initialize ChromaDB Client and Collection
client = chromadb.Client()
collection = client.create_collection("pdf_collection")

# Step 3: Generate Embeddings for Text Chunks
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedder.encode(text_chunks)

# Step 4: Add Chunks and Embeddings to ChromaDB
for i, chunk in enumerate(text_chunks):
    collection.add(
        ids=[f"chunk_{i}"],  # Unique ID for each chunk
        documents=[chunk],
        embeddings=[embeddings[i].tolist()]  # Convert to list for storage
    )

# Step 5: Print success message with total chunks
print(f"Vector database for '{pdf_file}' has been successfully created with {len(text_chunks)} chunks!")
# Query the collection to verify the number of documents in the database
db_count = collection.count()
print(f"Total documents in the vector database: {db_count}")


