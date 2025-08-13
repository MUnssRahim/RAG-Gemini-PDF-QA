import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_community.vectorstores import Pinecone as LangPinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import streamlit as st
from pinecone import Pinecone  # <-- Using same as your earlier code

# Load environment variables
load_dotenv()

# Retrieve API keys from .env
GOOGLE_API_KEY = os.getenv("Google_API_KEY")
PINECONE_API_KEY = os.getenv("Pinecone_API_KEY")

# Configure Google GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone (like your first RAG code)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = ""

def getpdftext(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def texttochunk(text, chunk_size=10000, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def getconversation(prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'questions'])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def storetoPinecone(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="embedding-v1")
        LangPinecone.from_texts(text_chunks, embedding=embeddings, index_name=index_name)
    except Exception as e:
        raise

def userinput(userquestion, chain):
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-v1")
    try:
        vectorstore = LangPinecone.from_existing_index(index_name=index_name, embedding=embeddings)
        docs = vectorstore.similarity_search(userquestion)
        response = chain({"input_documents": docs, "question": userquestion}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error("Error with Pinecone index. Please upload a PDF file first.")

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":book:")
    st.header("Chat using Google's Gemini")

    prompt_template = "Given the following context: {context}\nAnswer the following questions: {questions}"
    chain = getconversation(prompt_template)

    with st.sidebar:
        st.title("Menu")
        st.write("Upload your PDF file")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            try:
                with st.spinner("Please wait..."):
                    rawtext = getpdftext(uploaded_file)
                    text_chunks = texttochunk(rawtext, chunk_size=400, chunk_overlap=40)
                    storetoPinecone(text_chunks)
                    st.success("Completion")
            except Exception as e:
                st.error(f"Error occurred in uploading: {e}")

    user_question = st.text_input("Enter your question:")
    if user_question:
        userinput(user_question, chain)

if __name__ == "__main__":
    main()
