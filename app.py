import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from urllib.parse import urlparse
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from tempfile import NamedTemporaryFile
import os
import datetime



# Function to generate a unique file name based on the current date and time
def get_unique_file_name():
    """
    Generate a unique file name based on the current date and time.

    Returns:
        str: A unique file name string.
    """
    return str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')


# Function to check if a string is a valid URL
def is_url(url):
    """
    Check if a string is a valid URL.

    Args:
        url (str): The URL to be checked.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Function to extract text from local PDFs and return a list of documents
def get_text_from_localPDFs(pdf_docs):
    """
    Extract text from local PDF files and return a list of documents.

    Args:
        pdf_docs (list): A list of uploaded PDF file objects.

    Returns:
        list: A list of documents containing extracted text.
    """
    documents = []
    for pdf in pdf_docs:
        bytes_data = pdf.read()
        with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
            tmp.write(bytes_data)                      # write data from the uploaded file into it
            pdf_reader = PyPDFLoader(tmp.name).load()        # <---- now it works!
            documents.extend(pdf_reader)
        os.remove(tmp.name)   
    return documents

# Function to extract text from PDFs hosted at URLs and return a list of documents
def get_pdf_text_fromURLs(URLs):
    """
    Extract text from PDF files hosted at URLs and return a list of documents.

    Args:
        URLs (list): A list of URLs pointing to PDF files.

    Returns:
        list: A list of documents containing extracted text.
    """
    documents = []
    for url in URLs:
        pdf_reader = PyPDFLoader(url)
        pages = pdf_reader.load_and_split()
        documents.extend(pages)
    return documents

# Function to split text into smaller chunks for processing
def get_text_chunks(text):
    """
    Split a large text into smaller chunks for processing.

    Args:
        text (str): The text to be split.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(text)
    return texts


# Function to generate a vector store from text chunks
def get_vectorstore(text_chunks):
    """
    Generate a vector store from a list of text chunks.

    Args:
        text_chunks (list): A list of text chunks.

    Returns:
        langchain.vectorstores.FAISS: A vector store created from the text chunks.
    """
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=[text.page_content for text in text_chunks], embedding=embeddings)
    return vectorstore

# Function to load a vector store from a local file
def load_vectorstore(path_to_vectorstore):
    """
    Load a vector store from a local file.

    Args:
        path_to_vectorstore (str): The path to the vector store file.

    Returns:
        langchain.vectorstores.FAISS: The loaded vector store.
    """
    embeddings = OpenAIEmbeddings()
    # #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.load_local(path_to_vectorstore, embeddings)
    return vectorstore

# Function to create a conversation chain for user interactions
def get_conversation_chain(vectorstore):
    """
    Create a conversation chain for user interactions.

    Args:
        vectorstore (langchain.vectorstores.FAISS): A vector store for retrieval.

    Returns:
        langchain.chains.ConversationalRetrievalChain: The conversation chain.
    """
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1024})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display chat responses
def handle_userinput(user_question):
    """
    Handle user input and display chat responses.

    Args:
        user_question (str): The user's question or input.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# Main function for running the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'text_chunk' not in st.session_state:
        st.session_state.text_chunk = []
    if 'newDatabase' not in st.session_state:
        st.session_state.newDatabase = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    # Streamlit UI elements
    st.header("Chat with multiple PDFs :books:")

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        st.write(f"Your prompt: {user_question}")
        handle_userinput(user_question)

    # Sidebar for uploading PDFs and creating a vector store
    with st.sidebar:
        st.subheader("Query your personal PDFs or URLs or a Mixture of both", divider='rainbow')
        col1, col2 = st.columns(2)

        with col1:
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process PDFs'", accept_multiple_files=True)
            
            if st.button("Process PDFs"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_text_from_localPDFs(pdf_docs)
                    
                    # get the text chunks from local storgae PDFs
                    st.session_state.text_chunk.extend(get_text_chunks(raw_text))
                    
                    
        with col2:
            text_input = st.text_input(
                "Enter PDFs URLs(e.g url1, url2 (use comma to distinguish b/w different URLs)) and click on 'Process URLs'"
            )

            if st.button("Process URLs"):
                if text_input:
                    st.write("You entered: ", text_input)
                    URLs = text_input.split(',')
                    for url in URLs:
                        if url is not None and is_url(url) == False:
                            st.write('This is not a valid URLs : ', url)
                            st.write('Please Upload valid URLs')

                    with st.spinner("Processing"):
                        # get pdf text from URLs
                        raw_text = get_pdf_text_fromURLs(URLs)

                        # get the text chunks
                        st.session_state.text_chunk.extend(get_text_chunks(raw_text))
        

        st.write('After processing URLs/PDFs, click on create database to generate a vectorstore')
        if st.button("Create Database"):
                with st.spinner("Processing"):
                    #st.write(st.session_state.text_chunk)

                    # create vector store
                    vectorstore = get_vectorstore(st.session_state.text_chunk)

                    #create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    # Save the new vectorstate as session state, will be useful when we need to save database
                    st.session_state.vectorstore = vectorstore

                    # set the state to save the newDatabase 
                    st.session_state.newDatabase = True
                    
                    st.write('Hurray!!! your personlized assistance is ready 	:robot_face: You can also save this vectorstore for future use ')


        if st.session_state.newDatabase:   
                if st.button('Save the newly created Database'):
                    with st.spinner("Processing"):
                        fname = get_unique_file_name()
                        # use the session_state for vectorstore
                        st.session_state.vectorstore.save_local(
                            os.path.join( 
                                os.getcwd(),
                                'VectorDataBase',
                                fname
                        ))
                        st.write(f'Saved the database at : VectorDataBase/{fname}')


        st.subheader("Load pre-existing Vector-Database for personalized Query",  divider='rainbow')
        path_to_vectorstore = st.text_input('Specify path to Vector-Database here')
        if st.button("Load Database"):
            with st.spinner("Processing"):
                # load vector store
                st.session_state.vectorstore = load_vectorstore(path_to_vectorstore)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

                st.write('Hurray!!! your personlized assistance is ready 	:robot_face:')


        expander1 = st.expander("About APP")
        expander1.write("""
        Chat with Multiple PDFs is a user-friendly web application that simplifies the process of interacting with PDF documents using artificial intelligence.
        It's designed to assist you in extracting valuable information from your PDFs and answering your questions.
        """)

        st.markdown('''
        The primary resource used to create the app:
                    
        - [streamlit](https://streamlit.io/)
        - [Langchain](https://docs.langchain.com/docs/)
        - [OpenAI](https://openai.com/)
        - [HuggingFace](https://huggingface.co/)

        ## About me:

        - [Linkedin](https://www.linkedin.com/in/pankaj-chouhan/)
        
        ''')


        expander2 = st.expander("HuggingFace Warning")
        expander2.write("""
                        ### **Switching Between OpenAI and Huggingface Embeddings/LLMs**

                        - **Current Version and Cost:**
                        - The current version uses OpenAI embeddings ('text-embedding-ada-002') and OpenAI LLM ('gpt-3.5-turbo').
                        - Using OpenAI embeddings incurs a minimal cost, which can add up when dealing with large corpuses.

                        - **Options for Handling Large Corpus:**
                        - To address cost concerns with large corpuses, you have two options:
                            - Calculate the embedding for the entire corpus at once and save the vectorstore.
                            - Switch to Huggingface models for embedding and LLM.

                        - **Benefits of Huggingface:**
                        - One of the primary benefits of using Huggingface is that it allows you to run everything locally and is free of cost.

                        - **Switching to Huggingface:**
                        - To switch to Huggingface embeddings and LLM, you should:
                            - Comment out the OpenAI embedding/LLM.
                            - Uncomment the Huggingface embedding/LLM.

                        ### **Considerations When Using Huggingface**

                        - **Performance on Non-GPU Machines:**
                        - It's important to be aware that using Huggingface can lead to performance issues, especially on machines without GPUs. Running the app with Huggingface embeddings may be significantly slower in such cases.

                        - **Token Limitations:**
                        - Huggingface models have a relatively small token limit.
                        - This limitation becomes apparent when you query your model multiple times in sequence.
                        - In the case of LLM, it takes into account all past input and output and feeds that along with the new question to generate a response.
                        - Consequently, the token limit is reached quickly.
                        - OpenAI "gpt-3.5-turbo" has a limit of 4096 tokens, while most Huggingface models have a limit of approximately 1000 tokens.

                        - **Handling Token Limit Issues:**
                        - If you decide to use Huggingface and encounter token limit problems, there are two approaches to consider:
                            - Create smaller chunks of text by reducing the `chunk_size` parameter in the `get_text_chunks` function. However, note that this approach may result in poorer responses due to potential loss of semantic information.

                        - **Alternative Approach for Token Limit:**
                        - Another approach is to adjust the `map_reduce` as `chain_type` in the `ConversationBufferMemory` call.
                        - The default value is 'stuff.'
                        - `map_reduce` essentially splits the large text and creates a summary of each portion, which is further reduced to a concise summary. However, this approach can result in a loss of semantic information.
                        - On the other hand, 'stuff' shoves the entire conversation chain at once but can lead to reaching the token limit quickly.

                        - **Choose a Huggingface Model with a Large Token Limit:**
                        - When using Huggingface, it's advisable to select a model with a larger token limit to accommodate more extensive conversations and queries.
        """)


if __name__ == '__main__':
    main()
