# Code adapted from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps and https://github.com/aniketpotabatti/Gemini-PDF-Question-Answering-System

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Generate the embeddings for the vector store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Call the respective model
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_openai import ChatOpenAI

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import os
import time


load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.

    Args:
        pdf_docs: A list of PDF documents.

    Returns:
        A string containing the extracted text from all PDF documents.
    """

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits a large text into smaller chunks for efficient processing.

    Args:
        text: A string containing the text to be split.

    Returns:
        A list of text chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates a vector store from a list of text chunks.

    Args:
        text_chunks: A list of text chunks.

    Returns:
        A FAISS vector store.
    """

    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # text-embedding-3-small, text-embedding-3-large
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store_oa = FAISS.from_texts(text_chunks, embedding=openai_embeddings)
    vector_store_oa.save_local("openai_index")

    vector_store_gm = FAISS.from_texts(text_chunks, embedding=gemini_embeddings)
    vector_store_gm.save_local("gemini_index")

    return vector_store_oa, vector_store_gm

def get_conversional_chain():
    """
    Creates a conversational chain for question answering.

    Returns:
        A LangChain question-answering chain.
    """

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not available in the context, just say, "answer is not available in the context", don't provide the wrong answer.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    openai_model = ChatOpenAI(model="gpt-4o-mini")
    gemini_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3) # temp => predictable and repetitive vs random and creative
 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_oa = prompt | openai_model
    chain_gm = prompt | gemini_model
    return chain_oa, chain_gm


def generate_response(user_question, processed_pdf_text):
    """
    Processes user input and generates a response using the conversational chain,
    providing both the user's question and the processed PDF text as context.

    Args:
        user_question: The user's question.
        processed_pdf_text: The processed text extracted from the uploaded PDF files.

    Returns:
        The generated response from the conversational chain.
    """
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # text-embedding-3-small, text-embedding-3-large
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


    new_db_oa = FAISS.load_local("openai_index", openai_embeddings, allow_dangerous_deserialization=True)
    new_db_gm = FAISS.load_local("gemini_index", gemini_embeddings, allow_dangerous_deserialization=True)

    docs_oa = new_db_oa.similarity_search(user_question)
    docs_gm = new_db_gm.similarity_search(user_question)

    chain_oa, chain_gm = get_conversional_chain()

    # Combine user question and processed PDF text as context
    context = f"{processed_pdf_text}\n\nQuestion: {user_question}"

    response_oa = chain_oa.invoke({"input_documents": docs_oa, "question": user_question, "context": context})  
    response_gm = chain_gm.invoke({"input_documents": docs_gm, "question": user_question, "context": context})  

#    print(response)
    return response_oa.content, response_gm.content


def main():
    """
    Main function for the Streamlit app.
    """

    st.set_page_config("Chat With Multiple PDF files", page_icon=":books:")
    st.header("Chat with PDF files powered by GPT-4o-mini and Gemini-Pro :books: 🙋‍♂️")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("What's your next question 🔎❔")


    if user_question:
        if st.session_state.get("pdf_docs"):
            # store user message in the history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # get user response
            processed_pdf_text = get_pdf_text(st.session_state["pdf_docs"])
            response_oa, response_gm = generate_response(user_question, processed_pdf_text)
            
            def response_oa_stream():
                for word in response_oa.split(" "):
                    yield word + " "
                    time.sleep(0.03)

            def response_gm_stream():
                for word in response_gm.split(" "):
                    yield word + " "
                    time.sleep(0.03)

            # display user response
            with st.chat_message("openai"):    
                 st.write_stream(response_oa_stream)
            with st.chat_message("gemini"):    
                 st.write_stream(response_gm_stream)

            # store system response in the history
            st.session_state.messages.append({"role": "openai", "content": response_oa, "avatar":"🤖"})
            st.session_state.messages.append({"role": "gemini", "content": response_gm, "avatar":"💻"})

        else:
            st.error("Please upload PDF files first.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Files & Click Submit to Proceed", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                text_chunks = []
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                text_chunks = get_text_chunks(raw_text)
                vector_store_oa, vector_store_gm = get_vector_store(text_chunks)
                chain_oa, chain_gm = get_conversional_chain()
                st.session_state["pdf_docs"] = pdf_docs
                st.session_state["text_chunks"] = text_chunks
                st.session_state["vector_store_oa"] = vector_store_oa
                st.session_state["vector_store_oa"] = vector_store_gm
                st.session_state["chain_oa"] = chain_oa
                st.session_state["chain_gm"] = chain_gm
                st.success("PDFs processed successfully!")

        if st.button("Reset"):
            st.session_state["pdf_docs"] = []
            st.session_state["text_chunks"] = []
            st.session_state["vector_store_oa"] = None
            st.session_state["vector_store_oa"] = None
            st.session_state["chain_oa"] = None
            st.session_state["chain_gm"] = None
            st.rerun()

        if st.session_state.get("pdf_docs"):
            st.subheader("Uploaded Files:")
            for i, pdf_doc in enumerate(st.session_state["pdf_docs"]):
                st.write(f"{i+1}. {pdf_doc.name}")


if __name__ == "__main__":
    main()