from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
# Load PDFs
loader1 = PyPDFLoader("data/API Documentation Partial.pdf")
loader2 = PyPDFLoader("data/How to use DeepInfra MetaLLama.pdf")


loader3 = TextLoader("data/Pasted text(1).txt")
docs1 = loader1.load()
docs2 = loader2.load()
docs3 = loader3.load()

documents = docs1 + docs2 + docs3


# Sanity check
full_text = "\n".join([doc.page_content for doc in documents])

print("Total Characters:", len(full_text))
print("\nSample Text:\n")
print(full_text[:1000])

# BETTER CHUNKING
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

split_docs = splitter.split_documents(documents)

print("Total Chunks:", len(split_docs))

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory="vectorstore"
)

vectorstore.persist()

print("Vector DB Created Successfully")