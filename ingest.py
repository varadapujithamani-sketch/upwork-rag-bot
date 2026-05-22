from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


def build_vectorstore():

    # =========================
    # 1. LOAD DOCUMENTS
    # =========================

    pdf_loader = PyPDFLoader(
        "data/API Documentation Partial.pdf"
    )

    qa_loader = TextLoader(
        "data/Pasted text(1).txt"
    )

    pdf_docs = pdf_loader.load()
    qa_docs = qa_loader.load()

    documents = pdf_docs + qa_docs

    # =========================
    # 2. SANITY CHECK
    # =========================

    full_text = "\n".join(
        [doc.page_content for doc in documents]
    )

    print("\n===== SANITY CHECK =====\n")

    print("Total Character Count:")
    print(len(full_text))

    print("\nSample Text:\n")
    print(full_text[:1000])

    # =========================
    # 3. DOCUMENT CHUNKING
    # =========================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = splitter.split_documents(documents)

    print("\nTotal Chunks Created:")
    print(len(split_docs))

    # =========================
    # 4. EMBEDDING MODEL
    # =========================

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # =========================
    # 5. VECTOR DATABASE
    # =========================

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory="vectorstore"
    )

    vectorstore.persist()

    print("\nVectorstore Created Successfully")


if __name__ == "__main__":
    build_vectorstore()