from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def get_load_docs():
    """
    Load all PDFs in the data directory
    """
    loader = DirectoryLoader(
        path="./data",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,
    )
    docs = loader.load()
    return docs

def get_text_splitted_and_chunked(extract_document):
    """
    Split and chunk text into smaller segments.

    Parameters:
    - extract_document (str): The text document to be split and chunked.

    Returns:
    - text_chunks (list): A list containing the split and chunked text segments.

    Example:
    >>> document = "Lorem ipsum dolor sit amet, consectetur adipiscing elit..."
    >>> chunks = get_text_splitted_and_chunked(document)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        is_separator_regex=True,
    )
    text_chunks = text_splitter.split_documents(extract_document)
    return text_chunks

def get_embeddings():
    """
    Initializes and returns an instance of Hugging Face Embeddings.

    Returns:
    HuggingFaceEmbeddings: An instance of Hugging Face Embeddings
    initialized with the specified model name.
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
