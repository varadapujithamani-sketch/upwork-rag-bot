from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore():

    loader1 = PyPDFLoader("data/API Documentation Partial.pdf")
    loader2 = PyPDFLoader("data/How to use DeepInfra MetaLLama.pdf")
    loader3 = TextLoader("data/Pasted text(1).txt")

    docs1 = loader1.load()
    docs2 = loader2.load()
    docs3 = loader3.load()

    documents = docs1 + docs2 + docs3

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory="vectorstore"
    )

    vectorstore.persist()

    print("Vectorstore created successfully")


if __name__ == "__main__":
    build_vectorstore()