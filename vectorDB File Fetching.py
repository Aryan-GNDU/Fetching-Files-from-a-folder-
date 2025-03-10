import os
import dotenv
import logging
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pinecone import Pinecone, ServerlessSpec
import torch
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------
# Load environment variables (Pinecone API key etc.)
dotenv.load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "us-east-1")

# --------------------
# File and index settings (edit these as needed)
PDF_FOLDER = "pdf"  # Folder with PDFs
INDEX_NAME = "hybrid-search-langchain-pinecone"
EMBEDDING_DIM = 384  # Matches the all-MiniLM-L6-v2 model's output dimension

# --------------------
# 1. Load and split PDFs
logger.info(f"Loading PDFs from folder: {PDF_FOLDER}")
if not os.path.exists(PDF_FOLDER):
    logger.info(f"Creating PDF folder: {PDF_FOLDER}")
    os.makedirs(PDF_FOLDER)
    logger.warning(f"Please add your PDF files to the '{PDF_FOLDER}' directory and run the script again.")
    exit(0)

# Check if PDFs exist
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
if not pdf_files:
    logger.warning(f"No PDF files found in '{PDF_FOLDER}' directory. Please add PDFs and run again.")
    exit(0)

logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")

# Process each PDF file individually for better debugging
all_docs = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    logger.info(f"Processing PDF: {pdf_file}")
    
    try:
        # Load individual PDF
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        logger.info(f"Loaded {len(pdf_docs)} pages from {pdf_file}")
        
        # Add better source metadata
        for doc in pdf_docs:
            doc.metadata["source"] = pdf_file
            doc.metadata["filename"] = pdf_file
        
        all_docs.extend(pdf_docs)
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_file}: {e}")

logger.info(f"Total pages loaded from all PDFs: {len(all_docs)}")

# Split documents
logger.info("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)
logger.info(f"Created {len(docs)} document chunks")

# --------------------
# 2. Initialize embeddings
logger.info("Initializing embeddings...")
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# --------------------
# 3. Setup Pinecone and upsert vectors
logger.info("Initializing Pinecone...")
# Initialize Pinecone client using the latest version
pc = Pinecone(api_key=api_key)

# Check available indices
try:
    available_indexes = [index.name for index in pc.list_indexes()]
    logger.info(f"Available Pinecone indexes: {available_indexes}")
except Exception as e:
    logger.error(f"Error listing Pinecone indexes: {e}")
    available_indexes = []

# Create index if it does not exist
if INDEX_NAME not in available_indexes:
    logger.info(f"Creating Pinecone index: {INDEX_NAME}")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=PINECONE_API_ENV)
        )
        logger.info(f"Created Pinecone index: {INDEX_NAME}")
        # Wait for index to initialize
        time.sleep(10)
    except Exception as e:
        logger.error(f"Error creating Pinecone index: {e}")
        exit(1)
else:
    logger.info(f"Pinecone index '{INDEX_NAME}' already exists.")

# Connect to the index
try:
    index = pc.Index(INDEX_NAME)
    # Get index stats
    stats = index.describe_index_stats()
    logger.info(f"Current index stats: {stats}")
except Exception as e:
    logger.error(f"Error connecting to Pinecone index: {e}")
    exit(1)

# Ask if user wants to clear the index
clear_index = input("Do you want to clear the existing index before adding new documents? (y/n): ")
if clear_index.lower() == 'y':
    logger.info("Clearing index...")
    try:
        index.delete(delete_all=True)
        logger.info("Index cleared successfully")
        # Wait for deletion to complete
        time.sleep(5)
    except Exception as e:
        logger.error(f"Error clearing index: {e}")

# Upsert documents with metadata
logger.info("Upserting document vectors into Pinecone...")
batch_size = 100  # Process in batches to avoid memory issues

# Track which PDFs are being indexed
indexed_sources = set()

for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    vectors = []
    
    for j, doc in enumerate(batch):
        try:
            # Create a unique ID including source information
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            chunk_id = j % batch_size
            doc_id = f"{source.replace('.pdf', '')}_{page}_{chunk_id}_{i+j}"
            
            # Create embedding vector
            vector = embeddings.embed_query(doc.page_content)
            
            # Add source to tracking
            indexed_sources.add(source)
            
            # Create metadata
            metadata = {
                "text": doc.page_content,
                "source": source,
                "page": page,
                "doc_id": doc_id
            }
            
            # Add to batch
            vectors.append((doc_id, vector, metadata))
        except Exception as e:
            logger.error(f"Error creating vector for document {i+j}: {e}")
    
    # Upsert batch
    if vectors:
        try:
            index.upsert(vectors=vectors)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} with {len(vectors)} vectors")
        except Exception as e:
            logger.error(f"Error upserting batch to Pinecone: {e}")

logger.info(f"Indexed sources: {indexed_sources}")

# Verify indexed data
try:
    stats = index.describe_index_stats()
    logger.info(f"Index stats after upsert: {stats}")
    total_vectors = stats.get("total_vector_count", 0)
    logger.info(f"Total vectors in index: {total_vectors}")
except Exception as e:
    logger.error(f"Error getting index stats: {e}")

# --------------------
# 4. Initialize LangChain's Pinecone wrapper for retrieval
logger.info("Initializing LangChain Pinecone wrapper...")
docsearch = LC_Pinecone(index, embeddings, text_key="text")

# --------------------
# 5. Setup language model using HuggingFacePipeline
logger.info("Loading language model...")
# Try to load Llama 2 if available, otherwise fall back to a smaller model
model_id = "meta-llama/Llama-2-7b-chat-hf"  # Requires access token

try:
    # Check if CUDA is available for GPU acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Configure pipeline with appropriate parameters
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
except Exception as e:
    logger.error(f"Error loading Llama 2: {e}")
    logger.info("Falling back to a smaller, open-access model...")
    try:
        # Fallback to a smaller, open model that doesn't require access tokens
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Much smaller, open model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto" if device == "cuda" else None)
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            do_sample=True,
        )
    except Exception as e2:
        logger.error(f"Error loading fallback model: {e2}")
        logger.error("Please ensure you have access to the model or modify the script to use another available model.")
        exit(1)

# Create LangChain wrapper for the model
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# --------------------
# 6. Create a Retrieval QA Chain for retrieving documents
logger.info("Initializing document retrieval...")
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 most similar documents
)

# --------------------
# 7. Interactive search loop - Focus on sources only
logger.info("\nSearch your documents! Type 'exit' to quit.")
logger.info(f"Available documents: {list(indexed_sources)}")
print("\nSearch your documents! Type 'exit' to quit.")
print(f"Available documents: {', '.join(indexed_sources)}")

while True:
    query = input("\nEnter your question: ")
    if query.lower() in ("exit", "quit"):
        print("Exiting. Goodbye!")
        break

    try:
        # Perform direct similarity search
        results = docsearch.similarity_search(query, k=5)
        
        if not results:
            print("No relevant content found in the indexed documents.")
            continue
        
        # Display sources only
        print(f"\nFound information in the following sources:")
        
        # Group results by source
        sources_dict = {}
        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            if source not in sources_dict:
                sources_dict[source] = []
                
            sources_dict[source].append({
                "page": page,
                "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            })
        
        # Display grouped results
        for i, (source, pages) in enumerate(sources_dict.items()):
            print(f"\n{i+1}. Document: {source}")
            print(f"   Found on {len(pages)} page(s): {', '.join(str(p['page']) for p in pages)}")
            print(f"   Sample content: {pages[0]['content']}")
    
    except Exception as e:
        print(f"Error searching documents: {e}")
        logger.error(f"Error searching for '{query}': {e}")
        print("Please try a different query.")