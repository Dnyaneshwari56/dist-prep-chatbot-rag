import os
import json
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_COLLECTION_NAME = "disaster_prep"

class DocumentProcessor:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = 500  # Number of characters per chunk
        self.overlap = 50

    def chunk_text(self, text, chunk_size=None, overlap=None):
        """Split text into overlapping chunks based on character count"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                last_period = chunk.rfind('.')
                last_exclaim = chunk.rfind('!')
                last_question = chunk.rfind('?')
                last_sentence_end = max(last_period, last_exclaim, last_question)
                
                if last_sentence_end > chunk_size * 0.7:  # Only break if we're not losing too much
                    chunk = chunk[:last_sentence_end + 1]
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
                
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter out very short chunks

    def create_embeddings(self, texts):
        """Create embeddings for a list of texts"""
        if not texts:
            return []
        
        logger.info(f"Creating embeddings for {len(texts)} text chunks")
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        return embeddings

class QdrantManager:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        try:
            if qdrant_api_key:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=300)
            else:
                self.client = QdrantClient(url=qdrant_url, timeout=300)
            
            # Test connection
            self.client.get_collections()
            self.collection_name = VECTOR_COLLECTION_NAME
            self.use_local_storage = False
            logger.info(f"Connected to Qdrant at {qdrant_url}")
            
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant at {qdrant_url}: {e}")
            logger.info("Falling back to local file storage for embeddings")
            self.client = None
            self.use_local_storage = True
            self.local_storage_file = "data/embeddings_storage.json"
            self.embeddings_data = []

    def create_collection(self):
        """Create or recreate the collection"""
        if self.use_local_storage:
            logger.info("Using local file storage - no collection creation needed")
            # Clear existing local storage
            self.embeddings_data = []
            return
        
        try:
            # Check if collection exists and delete it
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            
            # Create new collection
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
            )
            logger.info(f"Collection {self.collection_name} created successfully")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def upload_chunks(self, doc, chunks, embeddings):
        """Upload document chunks with embeddings to Qdrant or local storage"""
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
            return
        
        if self.use_local_storage:
            self._upload_chunks_local(doc, chunks, embeddings)
        else:
            self._upload_chunks_qdrant(doc, chunks, embeddings)
    
    def _upload_chunks_local(self, doc, chunks, embeddings):
        """Upload chunks to local file storage"""
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a UUID for each chunk based on the URL and chunk index
            point_id_string = f"{doc['url']}#{i}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_string))
            
            point_data = {
                "id": point_id,
                "vector": embedding.tolist(),
                "payload": {
                    "content": chunk,
                    "title": doc["title"],
                    "source": doc["source"],
                    "scraped_date": doc["scraped_date"],
                    "type": doc["type"],
                    "url": doc["url"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_id": point_id_string  # Keep original ID for reference
                }
            }
            self.embeddings_data.append(point_data)
        
        logger.info(f"Added {len(chunks)} chunks to local storage for document: {doc['title'][:50]}...")
    
    def _upload_chunks_qdrant(self, doc, chunks, embeddings):
        """Upload chunks to Qdrant"""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a UUID for each chunk based on the URL and chunk index
            point_id_string = f"{doc['url']}#{i}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_string))
            
            point = qmodels.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "content": chunk,
                    "title": doc["title"],
                    "source": doc["source"],
                    "scraped_date": doc["scraped_date"],
                    "type": doc["type"],
                    "url": doc["url"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_id": point_id_string  # Keep original ID for reference
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Uploaded {len(points)} chunks for document: {doc['title'][:50]}...")
        except Exception as e:
            logger.error(f"Error uploading chunks: {e}")
            raise
    
    def save_local_storage(self):
        """Save embeddings data to local file"""
        if self.use_local_storage and self.embeddings_data:
            os.makedirs("data", exist_ok=True)
            with open(self.local_storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.embeddings_data)} embeddings to {self.local_storage_file}")

def main():
    """Main function to process documents and create embeddings"""
    # Check if scraped data exists
    data_file = "data/scraped_disaster_prep_data.json"
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please run scraper.py first to collect disaster preparedness data")
        return
    
    # Load and process documents
    logger.info(f"Loading documents from {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Initialize processor and Qdrant manager
    processor = DocumentProcessor(model_name=EMBEDDING_MODEL_NAME)
    qdrant_manager = QdrantManager()

    # Initialize Qdrant collection
    qdrant_manager.create_collection()

    # Process and upload documents
    total_chunks = 0
    for i, doc in enumerate(documents, 1):
        logger.info(f"Processing document {i}/{len(documents)}: {doc['title'][:50]}...")
        
        chunks = processor.chunk_text(doc["content"])
        if not chunks:
            logger.warning(f"No chunks created for document: {doc['title']}")
            continue
            
        chunk_embeddings = processor.create_embeddings(chunks)
        qdrant_manager.upload_chunks(doc, chunks, chunk_embeddings)
        
        total_chunks += len(chunks)
        logger.info(f"Processed {len(chunks)} chunks from document {i}")

    logger.info(f"Processing completed! Total chunks uploaded: {total_chunks}")
    print(f"\nProcessing completed!")
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks uploaded to Qdrant: {total_chunks}")
    print(f"Collection name: {VECTOR_COLLECTION_NAME}")

if __name__ == "__main__":
    main()
