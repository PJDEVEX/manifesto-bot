from langchain_core.prompts import PromptTemplate

def prompt(question, context):
    """
    Creates a prompt template for generating responses based on 
    presidential election manifestos.

    Parameters:
    - question (str): The specific question asked by the voter.
    - context (str): The context or information related to the manifestos.

    Returns:
    - PromptTemplate: A formatted prompt template.
    """
    prompt_template = f"""
    You are an honest, unbiased, and knowledgeable assistant and 
    reviewer who helps voters to assist them in answering critically 
    on the question {question}, they ask based on the manifestos of presidential 
    candidates of the upcoming presidential elections in Sri Lanka. 

    Your answers should be based on the manifestos of the presidential 
    candidates and should not be biased or influenced by any political 
    party or individual. 

    You should provide the complete, concise and answer with high readability 
    score within the given token limit. 

    Context: {context} 

    Response:
    """

    return PromptTemplate.from_template(prompt_template)