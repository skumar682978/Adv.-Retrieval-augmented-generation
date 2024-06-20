# Adv.-Retrieval-augmented-generation

This Streamlit application allows users to upload a PDF file, ask questions about its content, and receive answers using a language model. Here’s a short summary of the code:
1.	Setup and Imports:
-	Import necessary libraries including os, streamlit, and dotenv.
-	Load environment variables using load_dotenv().
-	Import and initialize the OpenAI GPT-4 model from langchain_openai using an API key.
  
2.	Define the rag Function:
-	Save the uploaded PDF to a temporary file and load it using PyMuPDFLoader.
-	Split the document into smaller chunks using RecursiveCharacterTextSplitter.
-	Create document embeddings with OpenAIEmbeddings and store them in a FAISS vector store.
-	Set up a retriever to fetch relevant document chunks based on the query.
-	Use CohereRerank to re-rank retrieved chunks and set up a ContextualCompressionRetriever.
-	Define a prompt template for the Q&A assistant.
-	Create a chain that combines the retriever, prompt, model, and output parser to process the query.
-	Execute the chain with the provided question and display the response using streamlit.
  
3.	Streamlit Interface:
-	Create a Streamlit application with a title "Talk to your PDF".
-	Add a sidebar for PDF file upload.
-	Provide an input field for the user's query.
-	On button click, call the rag function to process the query and display the result
  
The app integrates PDF handling, document chunking, embedding, retrieval, and a language model to answer user questions about the content of the uploaded PDF.

