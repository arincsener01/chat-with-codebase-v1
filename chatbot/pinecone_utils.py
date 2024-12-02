import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "codebase-vectors"  # Ensure this name is valid
VECTOR_DIMENSION = 384  # Match the dimension of your embeddings

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = (
    "codebase-vectors"  # Ensure this name is valid (lowercase alphanumeric + hyphens)
)
VECTOR_DIMENSION = 384  # Adjust to your embedding model's output dimension

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def connect_pinecone():
    """Ensure the Pinecone index exists and return the index object."""
    # Retrieve the list of existing indexes
    existing_indexes = pinecone_client.list_indexes().names()
    print(f"Existing indexes: {existing_indexes}")

    if INDEX_NAME not in existing_indexes:
        # Create the index if it doesn't exist
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",  # Ensure this region matches your Pinecone plan
            ),
        )
        print(f"Index '{INDEX_NAME}' created.")

    # Retrieve the index object
    index = pinecone_client.Index(INDEX_NAME)
    print(f"Connected to index: {index}")
    return index


def insert_vectors(vectors, metadata):
    """Insert vectors and metadata into Pinecone."""
    index = connect_pinecone()

    # Ensure metadata values are strings or valid Pinecone types
    valid_metadata = [
        {
            key: (
                str(value)
                if not isinstance(value, (str, int, float, bool, list))
                else value
            )
            for key, value in meta.items()
        }
        for meta in metadata
    ]

    # Prepare the data for upsert
    to_upsert = [
        (str(i), vector, meta)
        for i, (vector, meta) in enumerate(zip(vectors, valid_metadata))
    ]

    # Upsert the data into Pinecone
    index.upsert(to_upsert)


def search_vectors(query_vector, top_k=5):
    """Search for similar vectors in Pinecone."""
    index = connect_pinecone()

    # Query the index
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Extract matches from the query response
    matches = results.matches
    print("Matches:", matches)  # Debugging: Inspect matches

    # Format the results for easier handling
    formatted_results = [
        {"id": match.id, "score": match.score, "metadata": match.metadata}
        for match in matches
    ]

    return formatted_results
