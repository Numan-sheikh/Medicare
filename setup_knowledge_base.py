import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

DB_PATH = 'chroma_who_guidelines'
PDF_PATH = os.path.join('data', 'WHO Model List of Essential Medicines.pdf')

def setup_knowledge_base():
    print(f"Setting up WHO Knowledge Base from: {PDF_PATH}")
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: Could not find {PDF_PATH}")
        return False
        
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    print(f"Found {len(documents)} pages. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    
    print(f"Created {len(docs)} text chunks. Generating embeddings...")
    embedding_model = OllamaEmbeddings(model="llama3.2")
    
    # Clear old db if exists
    if os.path.exists(DB_PATH):
        print(f"Clearing old database at {DB_PATH}...")
        shutil.rmtree(DB_PATH)
        
    print("Building Chroma Vector Database...")
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embedding_model, 
        persist_directory=DB_PATH,
        collection_name="who_guidelines"
    )
    
    print("✅ WHO Knowledge Base setup complete!")
    print(f"Database saved to: {DB_PATH}")
    return True

if __name__ == '__main__':
    setup_knowledge_base()
