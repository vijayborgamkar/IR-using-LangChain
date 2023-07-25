import streamlit as st
import os
import tempfile
import pickle
import warnings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import shutil

# Ignore warnings
warnings.filterwarnings("ignore")


class Document:
    def __init__(self, page_content, source, page):
        self.page_content = page_content
        self.source = source
        self.page = page
        self.metadata = {'source': source, 'page': page}


# Load Multiple files from Directory
# @st.cache(allow_output_mutation=True)
# def load_documents(uploaded_files):
#     documents = []
#
#     for uploaded_file in uploaded_files:
#         if uploaded_file.type == "application/pdf":
#             doc_content = uploaded_file.read()
#             documents.append(Document(content=doc_content, metadata={}))
#
#     return documents


@st.cache(allow_output_mutation=True)
def load_documents(uploaded_files):
    # Create a temporary directory to store the uploaded files
    with tempfile.TemporaryDirectory() as tmp_dir:
        documents = []

        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                # Save the uploaded file to the temporary directory
                with open(os.path.join(tmp_dir, uploaded_file.name), 'wb') as f:
                    f.write(uploaded_file.getbuffer())

        # Now that all the files are saved in tmp_dir, use DirectoryLoader to load the documents
        loader = DirectoryLoader(tmp_dir, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents.extend(loader.load())

    return documents


# Store embeddings to FAISS Vector Store
def store_embeddings(documents, embeddings, store_name, path):
    vector_store = FAISS.from_documents(documents, embeddings)
    with open(os.path.join(path, f"{store_name}.pkl"), "wb") as f:
        pickle.dump(vector_store, f)


# Get Embeddings for the Documents
@st.cache(allow_output_mutation=True)
def get_embeddings(texts):
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                          model_kwargs={"device": "cpu"})
    embeddings = FAISS.from_documents(texts, instructor_embeddings)
    return embeddings


# Load or compute embeddings
def load_embeddings(store_name, path):
    with open(os.path.join(path, f"{store_name}.pkl"), "rb") as f:
        vector_store = pickle.load(f)
    return vector_store


# Retrieve relevant documents based on user input
def retrieve_documents(query, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return docs


# Streamlit app
def main():
    st.title("PDF Document Search")

    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    if uploaded_files:
        # Load the documents
        documents = load_documents(uploaded_files)

        # Ensure the documents are not empty
        if documents:
            # Divide and Conquer
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # Get Embeddings for the Documents
            embeddings = get_embeddings(texts)

            # Load the FAISS Vector Store
            Embedding_store_path = "./db_store"
            Embedding_store_name = "faiss_store"

            if not os.path.exists(Embedding_store_path):
                os.makedirs(Embedding_store_path)

            store_embeddings(texts, embeddings, Embedding_store_name, Embedding_store_path)

            vector_store = load_embeddings(Embedding_store_name, Embedding_store_path)

            # User input
            query = st.text_input("Enter your search query:")

            # Search and display results
            if st.button("Search"):
                # Retrieve relevant documents based on user input
                docs = retrieve_documents(query, vector_store)
                for i, doc in enumerate(docs):
                    st.write("Page content:", doc.page_content)
                    st.write("Metadata:", doc.metadata)
                    st.write("Source:", doc.metadata['source'])
                    st.write("Page:", doc.metadata['page'])
                    download_button = st.button(f"Download PDF {i + 1}")
                    if download_button:
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_filename = temp_file.name
                            temp_file.write(doc.page_content.encode())
                        st.write(f"[Download PDF {i + 1}](data:application/pdf;base64,{temp_filename})")
        else:
            st.write("No documents uploaded.")
    else:
        st.write("Please upload some PDF files.")


if __name__ == "__main__":
    main()
