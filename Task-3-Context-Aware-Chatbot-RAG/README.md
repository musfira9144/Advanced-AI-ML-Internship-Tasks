# 🧠 Task 3: Context-Aware Chatbot (RAG with LangChain & Groq)

## 📌 Overview
This project implements a **context-aware conversational chatbot** using **Retrieval-Augmented Generation (RAG)**.  
The chatbot is capable of:
- Remembering past conversation context
- Retrieving relevant information from a custom knowledge base
- Generating accurate responses using a Large Language Model (LLM)

This task demonstrates **real-world LLM system design**, combining vector databases, embeddings, memory, and LLM inference.

---

## 🎯 Objectives
- Build a conversational chatbot with memory
- Implement Retrieval-Augmented Generation (RAG)
- Store and retrieve documents using vector search
- Deploy the chatbot using **Streamlit**

---

## 🧰 Tech Stack
- **Python**
- **LangChain**
- **Groq LLM (LLaMA 3)**
- **FAISS** – Vector Database
- **Sentence-Transformers** – Embeddings
- **Streamlit** – Web UI
- **Google Colab** – Model development

---

## 🗂️ Project Structure
Task-3-Context-Aware-Chatbot-RAG/ 
├── app.py 
├── requirements.txt 
├── data/ 
│ └── knowledge.txt 
├── notebook/ 
│ └── RAG_Chatbot.ipynb 
├── README.md

---

## 🧠 How It Works (Architecture)
1. **Document Loading**  
   Custom text documents are loaded as the chatbot’s knowledge base.

2. **Text Chunking**  
   Documents are split into smaller chunks for better retrieval.

3. **Embedding Generation**  
   Each chunk is converted into vector embeddings using Sentence-Transformers.

4. **Vector Storage**  
   Embeddings are stored in a FAISS vector database.

5. **Context Memory**  
   Chat history is stored using LangChain’s conversation memory.

6. **LLM Response Generation**  
   Groq’s LLaMA 3 model generates responses using retrieved context + chat history.

---

## 🌐 Deployment

The chatbot is deployed using Streamlit.

## 🔗 Live App:
(Add your Streamlit Cloud link here once deployed)

---

## 📈 Skills Gained

- Conversational AI Development

- Retrieval-Augmented Generation (RAG)

- Vector Embeddings & Similarity Search

- LangChain Pipelines

- LLM Integration (Groq / LLaMA 3)

- Streamlit Deployment