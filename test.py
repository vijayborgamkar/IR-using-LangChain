import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

root_dir = "pdf files"

# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader(f'{root_dir}/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

unique_sources = set()

for doc in documents:
    source = doc.metadata['source']
    unique_sources.add(source)

num_unique_sources = len(unique_sources)

st.write("Number of unique sources:", num_unique_sources)

st.write()

st.write("Unique source names:")
for source_name in unique_sources:
    st.write(source_name)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

texts = text_splitter.split_documents(documents)


# Get Embeddings for the Documents
def store_embeddings(docs, embeddings, sotre_name, path):
    vectorStore = FAISS.from_documents(docs, embeddings)

    with open(f"{path}/faiss_{sotre_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)


def load_embeddings(sotre_name, path):
    with open(f"{path}/faiss_{sotre_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore


# HuggingFace Instructor Embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cpu"})

Embedding_store_path = f"{root_dir}/Embedding_store"

db_instructEmbedd = FAISS.from_documents(texts, instructor_embeddings)

retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})

# Get the question from user input
question = st.text_input("Enter your question:", "")
if st.button('Search'):
    docs = retriever.get_relevant_documents(question)
    if docs:
        content = docs[0].page_content
        metadata = docs[0].metadata
        source = docs[0].metadata['source']
        page_number = docs[0].metadata['page']

        st.write("Page content:", content)
        st.write("Metadata:", metadata)
        st.write("Source:", source)
        st.write("Page:", page_number)
    else:
        st.write("No relevant documents found.")
