# 📘 Retrieval-Augmented Generation (RAG) PDF QA System

A lightweight Retrieval-Augmented Generation (RAG) system designed to answer queries based on NCERT textbook PDFs using transformer embeddings and large language models (LLMs). Optimized to run on home computers with minimal resources.

---

## ✨ Features

- PDF-based question answering with **semantic chunking**
- Fast document retrieval using **FAISS**
- Embeddings from **MPNet**, **MiniLM**, and **fine-tuned MiniLM**
- **Streamlit interface** for easy interaction and model comparison
- Fine-tuned using **Multiple Negatives Ranking (MNR)** loss
- Query handling through **OpenRouter** LLM API (e.g., Gemma, DeepSeek, LLaMA)
