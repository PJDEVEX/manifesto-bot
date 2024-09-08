import time
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone_index(api_key, cloud, region, index_name, dimension):
    """
    Initialize a Pinecone index and return the index name and the index object.

    Args:
        api_key (str): Pinecone API key.
        cloud (str): Pinecone cloud name.
        region (str): Pinecone region name.
        index_name (str): Name of the index to create.
        dimension (int): Dimension of the index.

    Returns:
        tuple: A tuple containing the index name and the Pinecone index object.
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # Define serverless specifications
    spec = ServerlessSpec(cloud=cloud, region=region)

    # Check if the index exists, delete if it does
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Create a new index
    pc.create_index(
        dimension=int(dimension),
        name=index_name,
        metric="cosine",
        spec=spec
    )

    # Wait for the index to be ready before connecting
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    # Return both the index name and the index object
    index = pc.Index(index_name)
    return index_name, index
