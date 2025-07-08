import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gc # For garbage collection

# --- Configuration ---
# Get the absolute path of the directory containing this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, 'vector_store')
FILTERED_COMPLAINTS_FILE = os.path.join(DATA_DIR, 'filtered_complaints.csv')
FAISS_INDEX_NAME = 'faiss_complaints_index' # Name for the FAISS index files

# Ensure the vector_store directory exists
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Embedding Model Configuration
# Using a good general-purpose sentence embedding model.
# 'sentence-transformers/all-MiniLM-L6-v2' is efficient and performs well.
# For higher accuracy, consider larger models like 'BAAI/bge-small-en-v1.5' or 'sentence-transformers/all-mpnet-base-v2'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'} # Use 'cuda' if you have a compatible GPU
ENCODE_KWARGS = {'normalize_embeddings': True} # Normalize embeddings for better similarity search

# Text Splitter Configuration
CHUNK_SIZE = 500  # Max characters per chunk
CHUNK_OVERLAP = 50 # Overlap between chunks to maintain context

# --- 2.1: Load the cleaned and filtered dataset ---
print(f"Loading filtered data from {FILTERED_COMPLAINTS_FILE}...")
try:
    # Load with 'Complaint ID' as index, as it was saved that way.
    df = pd.read_csv(FILTERED_COMPLAINTS_FILE, index_col='Complaint ID')
    print(f"Successfully loaded {len(df)} records.")
except FileNotFoundError:
    print(f"ERROR: {FILTERED_COMPLAINTS_FILE} not found.")
    print(f"Please ensure Task 1 (EDA and Preprocessing) has been completed and the file exists.")
    exit()

# Ensure 'cleaned_narrative' column exists and handle potential NaNs
if 'cleaned_narrative' not in df.columns:
    print("ERROR: 'cleaned_narrative' column not found in the filtered data.")
    print("Please ensure Task 1 generated this column correctly.")
    exit()

# Drop rows where 'cleaned_narrative' is empty or NaN after loading and final checks
initial_rows = len(df)
df.dropna(subset=['cleaned_narrative'], inplace=True)
df = df[df['cleaned_narrative'].str.strip() != '']
if len(df) < initial_rows:
    print(f"Removed {initial_rows - len(df)} rows with empty/NaN cleaned narratives after loading.")

print(f"Total complaints with valid cleaned narratives for chunking: {len(df)}")


# --- 2.2: Chunk the text narratives ---
print("\n--- Chunking Text Narratives ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len, # Specifies that length should be measured by characters
    separators=["\n\n", "\n", ".", "!", "?", " ", ""] # Try to split by these in order
)

# Prepare documents for LangChain
# Each document will have the 'cleaned_narrative' as page_content
# and other relevant columns as metadata.
documents = []
for index, row in df.iterrows():
    # Use the 'cleaned_narrative' for the main content of the document
    page_content = row['cleaned_narrative']

    # Create metadata from other columns. Convert all metadata values to string for consistency.
    metadata = {
        col: str(row[col]) for col in df.columns if col != 'cleaned_narrative' and col != 'narrative_word_count' and col != 'cleaned_narrative_word_count'
    }
    # Add 'Complaint ID' to metadata for easy retrieval (it's the index in df)
    metadata['complaint_id'] = str(index) # Ensure it's a string

    # Create a LangChain Document object
    doc = Document(page_content=page_content, metadata=metadata)
    documents.append(doc)

print(f"Created {len(documents)} LangChain documents from filtered complaints.")

# Apply the text splitter to create chunks
print(f"Splitting documents into chunks (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
chunks = []
for doc in documents:
    # Use split_documents for Documents, which preserves metadata
    doc_chunks = text_splitter.split_documents([doc])
    chunks.extend(doc_chunks)

print(f"Total number of chunks created: {len(chunks)}")
# Display some chunk examples
if len(chunks) > 0:
    print("\n--- Example Chunks ---")
    for i, chunk in enumerate(chunks[:3]): # Display first 3 chunks
        print(f"Chunk {i+1}:")
        print(f"Content: {chunk.page_content[:200]}...") # Print first 200 chars
        print(f"Metadata: {chunk.metadata}")
        print("-" * 30)


# --- 2.3: Initialize the Embedding Model ---
print(f"\n--- Initializing Embedding Model: {EMBEDDING_MODEL_NAME} ---")
try:
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=EMBEDDING_MODEL_KWARTS,
        encode_kwargs=ENCODE_KWARGS
    )
    print("Embedding model loaded successfully.")
    # Test embedding to ensure it works
    _ = embeddings.embed_query("test embedding functionality")
    print("Embedding model test successful.")
except Exception as e:
    print(f"ERROR: Failed to load embedding model. Ensure 'sentence-transformers' and 'torch' are installed.")
    print(f"Error details: {e}")
    exit()

# --- 2.4: Create and Save the FAISS Vector Store ---
print("\n--- Creating and Saving FAISS Vector Store ---")

# Create the FAISS index from the chunks and embeddings
# This step can take a while depending on the number of chunks and model size
print("Generating embeddings and building FAISS index. This may take some time...")
try:
    db = FAISS.from_documents(chunks, embeddings)
    print("FAISS index created.")

    # Save the FAISS index locally
    faiss_path = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)
    db.save_local(faiss_path)
    print(f"FAISS index saved locally to: {faiss_path}")

except Exception as e:
    print(f"ERROR: Failed to create or save FAISS index.")
    print(f"Error details: {e}")
    exit()

# Optional: Load the FAISS index to verify (for debugging/testing)
print("\n--- Verifying FAISS Index Load ---")
try:
    # IMPORTANT: allow_dangerous_deserialization=True is required for loading FAISS indices
    # saved locally if they contain pickle-serialized components (like the embedding function or docstore).
    # Only set this to True if you trust the source of the saved index.
    loaded_db = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully for verification.")

    # Perform a dummy similarity search to confirm functionality
    query = "problem with unauthorized transactions on credit card"
    docs_with_scores = loaded_db.similarity_search_with_score(query, k=2)
    print(f"Test query: '{query}'")
    if docs_with_scores:
        print("Top 2 similar documents found during verification:")
        for doc, score in docs_with_scores:
            print(f"  Score: {score:.4f}")
            print(f"  Content: {doc.page_content[:150]}...")
            print(f"  Metadata: {doc.metadata}")
            print("-" * 20)
    else:
        print("No similar documents found during verification test.")

except Exception as e:
    print(f"ERROR: Failed to load or verify FAISS index.")
    print(f"Error details: {e}")

# Clean up memory
del df, documents, chunks, embeddings, db
gc.collect()

print("\n--- Task 2: Text Chunking, Embedding, and Vector Store Indexing Completed ---")
