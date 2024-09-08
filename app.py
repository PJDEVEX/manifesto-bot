import os
from dotenv import load_dotenv
import streamlit as st
from src.logger import logging
from src.utils import get_text_splitted_and_chunked, get_embeddings, get_load_docs
from src.prompt import create_contextualize_q_prompt, qa_prompt
from src.pineconedb import initialize_pinecone_index

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Setup Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.title("Presidential Election Manifestos Assistant")
st.write("This app helps voters to assist them in answering questions based on the manifestos of presidential candidates.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None and not st.session_state.document_uploaded:
    extract_document = get_load_docs()
    text_chunks = get_text_splitted_and_chunked(extract_document)
    
    # Initialize Pinecone index
    index_name, index = initialize_pinecone_index(
        api_key=os.getenv("PINECONE_API_KEY"),
        cloud=os.getenv("PINECONE_CLOUD"),
        region=os.getenv("PINECONE_REGION"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        dimension=os.getenv("PINECONE_DIMENSION")
    )

    embeddings = get_embeddings()

    vectorstore = PineconeVectorStore.from_texts(
        [t.page_content for t in text_chunks],
        embedding=embeddings,
        index_name=index_name
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    st.session_state.retriever = retriever
    st.session_state.document_uploaded = True
    st.write("Document uploaded and processed successfully.")

# Get the question from the user
if st.session_state.document_uploaded:
    question = st.text_input("Enter your question:")

    # Initialize OpenAI chat model
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_API_MODEL"),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS")),
        temperature=float(os.getenv("OPENAI_TEMPERATURE")),
    )

    if question:
        history_aware_retriever = create_history_aware_retriever(
            llm,
            st.session_state.retriever,
            create_contextualize_q_prompt(),
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt(
                question,
                st.session_state.retriever,
            ))

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            history_aware_retriever,
            lambda session_id: st.session_state.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Generate the answer
        answer = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "abc123"}},
        )

        if answer:
            # Post-process to add warmth
            response = f"{answer}\n\nI hope this answers your question! Feel free to ask anything else."
            st.write(response)
