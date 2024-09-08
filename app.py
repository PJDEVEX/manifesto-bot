import os
from dotenv import load_dotenv
import streamlit as st
import time
from src.logger import logging
from src.utils import get_text_splitted_and_chunked, get_embeddings, get_load_docs
from src.prompt import create_contextualize_q_prompt, qa_prompt
from src.pineconedb import initialize_pinecone_index

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
OPENAI_MAX_TOKENS = os.getenv("OPENAI_MAX_TOKENS")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_DIMENSION = os.getenv("PINECONE_DIMENSION")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['OPENAI_API_MODEL'] = OPENAI_API_MODEL
os.environ['OPENAI_MAX_TOKENS'] = OPENAI_MAX_TOKENS
os.environ['OPENAI_TEMPERATURE'] = OPENAI_TEMPERATURE
os.environ['EMBEDDING_MODEL_NAME'] = EMBEDDING_MODEL_NAME
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['PINECONE_INDEX_NAME'] = PINECONE_INDEX_NAME
os.environ['PINECONE_CLOUD'] = PINECONE_CLOUD
os.environ['PINECONE_REGION'] = PINECONE_REGION
os.environ['PINECONE_DIMENSION'] = PINECONE_DIMENSION

st.title("Presidential Election Manifestos Assistant")
st.write("This app helps voters to assist them in answering critically on the questions they ask based on the manifestos of presidential candidates of the upcoming presidential elections in Sri Lanka.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Get the question from the user
question = st.text_input("Enter your question:")

# submit button
submit = st.button("Submit")

# Initialize variables
extract_document = None
text_chunks = None
vectorstore = None

# Load the documents and split them into chunks
if uploaded_file is not None:
    for file in os.listdir("./data"):
        if file.endswith(".pdf"):
            extract_document = get_load_docs()
            st.write(f"Document loaded: {file}")
        text_chunks = get_text_splitted_and_chunked(extract_document)

# Call embeddings
embeddings = get_embeddings()

# Initialize Pinecone index
index_name, index = initialize_pinecone_index(
    api_key=PINECONE_API_KEY,
    cloud=PINECONE_CLOUD,
    region=PINECONE_REGION,
    index_name=PINECONE_INDEX_NAME,
    dimension=PINECONE_DIMENSION
)

if text_chunks is not None:
    vectorstore = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)


