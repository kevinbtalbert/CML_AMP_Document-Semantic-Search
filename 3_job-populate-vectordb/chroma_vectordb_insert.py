import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# load the document and split it into chunks

persistent_client = chromadb.PersistentClient("./chroma_db")
collection = persistent_client.get_or_create_collection(COLLECTION_NAME)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

# Define the path to your documents folder
folder_path = '/home/cdsw/data'

# List all PDF files in the folder
pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

for pdf_file in pdf_files:
    print(pdf_file)
    loader = PyPDFLoader(folder_path + "/" + pdf_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    # docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    print("Loaded PDF document " + pdf_file + " successfully to Pinecone index " + COLLECTION_NAME)

    # load it into Chroma
    #db = Chroma.from_documents(docs, embedding_function)
    # save to disk
    db = Chroma.from_documents(docs, embedding_function, client=langchain_chroma, collection_name=COLLECTION_NAME, persist_directory="./chroma_db")
