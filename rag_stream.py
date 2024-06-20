import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Load llm
from langchain_openai import ChatOpenAI

model = ChatOpenAI(name="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))

def rag(question:str, pdf):
    # Loading documents
    from langchain_community.document_loaders import PyMuPDFLoader 

    with open("temp.pdf", "wb") as temp_file:
        temp_file.write(pdf.getbuffer())
    
    # Load the PDF file using PyMuPDFLoader
    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()
    

    # Doc splitting 
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    # Doc embedding 
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents=documents, embedding=embeddings)

    # Setup retriver
    retriever = vector.as_retriever(top_k=10)

    # Re-rank the retrived chuncks
    from langchain.retrievers import  ContextualCompressionRetriever
    from langchain_cohere import CohereRerank


    compressor = CohereRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )


    # Prompt setup
    from langchain_core.prompts import ChatPromptTemplate

    template = """
                You are a Q&A assistant, You will refer the context provided and answer the question. 
                If you dont know the answer , reply that you dont know the answer:{context}
                Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)

    # Chain
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    rag_chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
        |prompt
        |model
        |StrOutputParser()
    )

    # Querry
    response = rag_chain.invoke(question)
    st.write(response)
    print(response)

# Streamlit

def main():
    st.title("Talk to your PDF")

    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    st.subheader("Enter query:")
    user_query = st.text_input("Your query")


    if uploaded_file is not None:
        if user_query is not None:
            if st.button("process"):
                rag(user_query,uploaded_file)

    

if __name__ == "__main__":
    main()







