import os
import streamlit as st

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Context-Aware RAG Chatbot")
st.title("🤖 Context-Aware Chatbot (RAG)")

# -----------------------------
# Set Groq API Key
# -----------------------------
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

# -----------------------------
# Load and Process Documents
# -----------------------------
@st.cache_resource
def load_rag_pipeline():
    loader = TextLoader("Task-3-Context-Aware-Chatbot-RAG/data/knowledge.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    return chatbot

chatbot = load_rag_pipeline()

# -----------------------------
# Chat Interface
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask a question:")

if user_input:
    response = chatbot.invoke({"question": user_input})
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response["answer"]))

for sender, message in st.session_state.messages:
    st.markdown(f"**{sender}:** {message}")

