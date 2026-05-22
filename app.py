import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="Upwork API Support Bot")

st.title("Upwork API Technical Support Bot")

question = st.text_input("Ask your question")

if question:

    answer, docs, latency = ask_question(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for i, doc in enumerate(docs):
        st.write(f"Source {i+1}:")
        st.write(doc.page_content[:300])
        
    st.subheader("Latency")
    st.write(f"{latency} seconds")