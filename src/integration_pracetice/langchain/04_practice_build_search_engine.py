from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embedding=embeddings)

file_path = "/Users/mu/Desktop/langchain-practice/doc/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,chunk_overlap=200,add_start_index=True
)

all_splits = text_splitter.split_documents(loader.load())

ids = vector_store.add_documents(all_splits)


results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")

print(results[0])