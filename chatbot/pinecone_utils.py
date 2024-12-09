import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "codebase-vectors"  # Ensure this name is valid
VECTOR_DIMENSION = 384  # Match the dimension of your embeddings

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def connect_pinecone():
    if not hasattr(connect_pinecone, "index"):
        # Connect and cache the index
        existing_indexes = pinecone_client.list_indexes().names()
        if INDEX_NAME not in existing_indexes:
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        connect_pinecone.index = pinecone_client.Index(INDEX_NAME)
    return connect_pinecone.index


def insert_vectors(vectors, metadata, namespace):
    """Insert vectors and metadata into Pinecone."""
    index = connect_pinecone()
    to_upsert = [
        (str(i), vector, meta)
        for i, (vector, meta) in enumerate(zip(vectors, metadata))
    ]
    index.upsert(to_upsert, namespace=namespace)


def search_vectors(query_vector, top_k=5, namespace=None):
    """Search for similar vectors in Pinecone."""
    index = connect_pinecone()
    results = index.query(
        vector=query_vector, top_k=top_k, namespace=namespace, include_metadata=True
    )
    return results
