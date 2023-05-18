import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]

def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    # txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    # pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    # csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    # all_files = txt_files + pdf_files + csv_files
    # return [load_single_document(file_path) for file_path in all_files]
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return [load_single_document(file_path) for file_path in all_files if file_path.endswith((".txt", ".pdf", ".csv"))]

def main():
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')       
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    #model_n_ctx = int(os.environ.get('MODEL_N_CTX'))
    chunk_size = int(os.environ.get('CHUNK_SIZE'))
    chunk_overlap = int(os.environ.get('CHUNK_OVERLAP'))
    
    # Load document and split in chunks
    # for root, dirs, files in os.walk("source_documents"):
    #     for file in files:
    #         if file.endswith(".txt"):
    #             loader = TextLoader(os.path.join(root, file), encoding="utf8")
    #         elif file.endswith(".pdf"):
    #             loader = PDFMinerLoader(os.path.join(root, file))
    #         elif file.endswith(".csv"):
    #             loader = CSVLoader(os.path.join(root, file))
    # documents = loader.load()
    
     # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    try:
        documents = load_documents(source_directory)
        
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        texts = text_splitter.split_documents(documents)
        print(f"Loaded {len(documents)} documents from {source_directory}")
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each, overlap {chunk_overlap} tokens)")
        
        # Create embeddings
        llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
        # Create and store locally vectorstore
        db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
        db.persist()
        db = None # Free memory
        print("Vectorstore created and stored successfully.")
        
    except FileNotFoundError:
        print(f"Directory {source_directory} not found!")
        exit
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
