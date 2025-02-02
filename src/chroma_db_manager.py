from chromadb import Client
from chromadb.config import Settings
import os


class ChromaDBManager:
    """
    A manager class to handle Chroma database instances for mandatory and voluntary memory storage.
    """

    def __init__(self, base_path="data/chroma_dbs"):
        """
        Initialize the ChromaDBManager with a base directory for database storage.
        """
        self.base_path = base_path

    def _get_db_path(self, thread_id: str, memory_type: str):
        """
        Construct the path for a Chroma database.
        """
        return os.path.join(self.base_path, f"{thread_id}_{memory_type}")

    def get_chroma_instance(self, thread_id: str, memory_type: str):
        """
        Retrieve or initialize a Chroma database instance for a specific thread_id and memory type.
        """
        db_path = self._get_db_path(thread_id, memory_type)
        os.makedirs(db_path, exist_ok=True)

        # Initialize Chroma Client
        client = Client(Settings(persist_directory=db_path))

        # Create or retrieve a collection
        collection_name = f"{thread_id}_{memory_type}_collection"
        collection = client.get_or_create_collection(name=collection_name)

        return collection

    def store_interaction(self, collection, text: str, embedding: list, doc_id: str):
        """
        Store an interaction in the Chroma collection.
        """
        try:
            collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[doc_id],
            )
        except Exception as e:
            print(f"Error storing interaction in ChromaDB: {e}")
            raise

    def query_memory(self, collection, query_embedding: list, k: int = 5):
        """
        Query the Chroma collection for relevant documents.
        """
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )
            return results
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            raise