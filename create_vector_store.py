# create_vector_store.py

from utils import IPCEmbeddingManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the embedding manager
    logger.info("Initializing embedding manager...")
    manager = IPCEmbeddingManager()
    
    # Create embeddings and save them
    logger.info("Creating embeddings...")
    manager.create_embeddings()
    
    logger.info("Saving vector store...")
    manager.save_vector_store()
    
    logger.info("Vector store created and saved successfully!")

if __name__ == "__main__":
    main() 