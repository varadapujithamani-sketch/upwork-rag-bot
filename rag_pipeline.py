import os
import time
import requests

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

import streamlit as st

API_KEY = st.secrets["DEEPINFRA_API_KEY"]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
import os
import ingest

if not os.path.exists("vectorstore"):
    ingest.build_vectorstore()

# Load vector DB
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

SYSTEM_PROMPT = """
You are a Senior Upwork API Consultant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:
"I'm sorry, but the provided documentation does not contain that information."

Do not hallucinate.
Do not assume information.
"""


def retrieve_docs(query):

    semantic_docs = vectorstore.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=20
    )

    keyword_docs = keyword_match(query, semantic_docs)

    if keyword_docs:
        return keyword_docs

    return semantic_docs


def generate_response(question, context):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {question}
                """
            }
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }

    response = requests.post(
        "https://api.deepinfra.com/v1/openai/chat/completions",
        headers=headers,
        json=payload
    )

    result = response.json()

    print("API RESPONSE:")
    print(result)

    if "choices" not in result:
        return f"API Error: {result}"

    return result["choices"][0]["message"]["content"]

def ask_question(question):

    start = time.time()

    docs = retrieve_docs(question)

    if not docs:
        return (
            "I'm sorry, but the provided documentation does not contain that information.",
            [],
            0
        )

    context = "\n\n".join([doc.page_content for doc in docs])

    answer = generate_response(question, context)

    latency = round(time.time() - start, 2)

    return answer, docs, latency
def keyword_match(query, docs):

    query_words = query.lower().split()

    matched_docs = []

    for doc in docs:

        text = doc.page_content.lower()

        score = sum(word in text for word in query_words)

        if score >= 2:
            matched_docs.append(doc)

    return matched_docs
