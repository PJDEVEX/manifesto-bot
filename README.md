# manifesto-bot

## Run the project locally 

* Clone the repo
`git clone https://github.com/username/manifesto-bot.git`

* setup venv
    
    * Create a virtual environment
        `source ~/anaconda3/bin/activate`
        `conda create --name mb python=3.10 -y`

    * Activate virtual environment
        `conda activate mb`

* Install dependencies
    `python3 setup.py install`

* Run requirements.txt
    `pip3 install -r requirements.txt`

* setup environment varialbes
- Please ref the sample .env file to setup your own env variables.

* Run streamlit app
`streamlit run app.py`

## Key challenge
- Could not generate the humanized chat conversation due to frequent \n
modifications to the `langchain` framework

## Reference
- [Langchain framwork](https://www.langchain.com/) - Framework for building generative ai \n
applications
- [OpenAI api](https://openai.com/api) - LLM model.
- [PineCone](https://docs.pinecone.io/home) - Vector database for storing vectorized data
- [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - \n
The model used for embedding extracted data
- [Streamlit]('https://pypi.org/project/streamlit/') - Framework for developing python based \n
applications