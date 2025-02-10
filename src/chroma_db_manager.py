import os
from chromadb import Client, PersistentClient
from chromadb.config import Settings

class ChromaDBManager:
    """
    A manager class to handle Chroma database instances for mandatory and voluntary memory storage.
    """
    def __init__(self, base_path="data/chroma_dbs"):
        """
        Initialize the ChromaDBManager with a base directory for database storage.
        """
        # Normalize the base path.
        self.base_path = os.path.abspath(base_path)
        # Cache clients keyed by the database path to avoid recreating them.
        self.clients = {}

    def _get_db_path(self, thread_id: str):
        """
        Construct the normalized persist directory for a given thread_id.
        We use only the thread_id so that all memory types for that thread share the same client.
        """
        return os.path.join(self.base_path, thread_id)

    def get_chroma_instance(self, thread_id: str, memory_type: str):
        """
        Retrieve or initialize a Chroma database instance for a specific thread_id.
        This returns a collection based on memory_type.
        """
        # Use the thread_id only to form the persist directory.
        db_path = self._get_db_path(thread_id)
        os.makedirs(db_path, exist_ok=True)

        # Use the normalized path as the key.
        if db_path in self.clients:
            client = self.clients[db_path]
        else:
            # client = Client(Settings(persist_directory=db_path))
            client = PersistentClient(path=db_path)
            self.clients[db_path] = client

        # Use memory_type to differentiate between collections.
        collection_name = f"{thread_id}_{memory_type}_collection"
        collection = client.get_or_create_collection(name=collection_name)
        return collection

    def store_interaction(self, collection, text: str, embedding: list, doc_id: str):
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
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )
            return results
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            raise
