from setuptools import setup, find_packages

setup(
    name='manifesto-bot',
    version='0.1',
    author='PJ',
    author_email='piyankara.jayadewa@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pypdf==4.3.1',
        'langchain-community==0.2.16',
        'langchain-openai==0.1.23',
        'sentence-transformers==3.0.1',
        'langchain==0.2.16',
        'langchain-pinecone==0.1.3',
        'streamlit==1.38.0',
        'pinecone-client==5.0.1',
        'numpy==1.26.4',
    ]
)
