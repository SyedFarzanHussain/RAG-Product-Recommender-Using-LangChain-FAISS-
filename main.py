from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()


# ---------------------------------------------------------
# 1. SETUP PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="ShopGenie AI", page_icon="🛍️")
st.title("🛍️ ShopGenie: Your AI Shopping Assistant")
st.write("Ask for any product, and I will find the best matches from our inventory!")

@st.cache_resource(show_spinner="Generating Embeddings...")
def load_data_and_vectorstore():
    # A. Load Data
    file_path = "products_dataset.csv"
    if not os.path.exists(file_path):
        st.error("Dataset not found! Please check the path.")
        st.stop()

    data = pd.read_csv(file_path)

    # B. Process Documents
    documents = []
    for index, row in data.iterrows():
        content = f"Title: {row['title']}\nDescription: {row['description']}"
        metadata = {'product_id': row['product_id'], 'title': row['title']}
        documents.append(Document(page_content=content, metadata=metadata))

    # C. Create Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store

@st.cache_resource(show_spinner="Loading Rag Chain")
def load_rag_chain(_vector_store):
    # A. Setup LLM (Using Zephyr-7B or DeepSeek)
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.2",
        task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)

    # B. Setup Retriever
    retriever = _vector_store.as_retriever(search_kwargs={"k": 4})

    # C. Setup Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a friendly and knowledgeable E-commerce Shopping Assistant.
    Your goal is to recommend products based strictly on the provided context.

    Instructions:
    1. Review the products in the "Context" section below.
    2. Recommend the best matching products to the user's question.
    3. Mention the product Title and explain WHY it fits their needs.
    4. If no product matches, politely say you don't have that item in stock.
    5. Be concise and use bullet points.

    Context:
    {context}

    User Question:
    {input}

    Answer:
    """)

    # D. Build Chains
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# ---------------------------------------------------------
# 4. INITIALIZE APP LOGIC
# ---------------------------------------------------------

with st.spinner("Loading AI Brain..."):
    vector_store = load_data_and_vectorstore()
    rag_chain = load_rag_chain(vector_store)

st.success("System Ready! ✅")

# ---------------------------------------------------------
# 5. CHAT INTERFACE
# ---------------------------------------------------------

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What are you looking for today?"):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Searching inventory..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)

            # Optional: Show Source Products in an Expander
            with st.expander("View Source Products"):
                for doc in response["context"]:
                    st.caption(f"**{doc.metadata['title']}** (ID: {doc.metadata['product_id']})")

    st.session_state.messages.append({"role": "assistant", "content": answer})

