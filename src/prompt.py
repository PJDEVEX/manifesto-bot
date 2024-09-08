from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

### Contextualize question
def create_contextualize_q_prompt():
    """
    Create a ChatPromptTemplate for contextualizing a user question
    based on the chat history, encouraging a conversational tone.
    
    Returns:
        ChatPromptTemplate: The prompt template for contextualizing questions.
    """
    contextualize_q_system_prompt = (
        "You are a friendly assistant. Given a chat history and the latest user question "
        "which might reference context in the chat history, formulate a standalone question "
        "in a conversational tone that can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it to be clear and friendly."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    return contextualize_q_prompt

def qa_prompt(question, context):
    """
    Creates a prompt template for generating responses based on 
    presidential election manifestos, with a conversational and friendly tone.

    Parameters:
    - question (str): The specific question asked by the voter.
    - context (str): The context or information related to the manifestos.

    Returns:
    - PromptTemplate: A formatted prompt template.
    """
    system_prompt = """
    You are an honest, unbiased, and knowledgeable assistant and 
    reviewer who helps voters by answering their questions based 
    on the manifestos of presidential candidates in the upcoming 
    elections in Sri Lanka.
    
    Use the following {context} to provide a clear and concise
    answer to the voter's question.

    Your answers should be based on the manifestos, unbiased, and 
    clear, but also friendly and conversational to engage with the 
    voter. Ensure your response is concise, with a high readability
    score within the given token limit. 

    Accordingly, you need to provide a clear and concise answer to
    the user's question: "{question}".
    
    answer:
    """
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

    return qa_prompt
