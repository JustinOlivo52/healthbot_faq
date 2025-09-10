# 🩺 Healthcare FAQ Bot (RAG Demo)

This is a simple **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** and **OpenAI**.  
It answers basic healthcare FAQ questions (hydration, fever, sleep, exercise) using a small knowledge base and provides **cited sources** for every answer.

⚠️ **Disclaimer:** This app is for educational purposes only and does **not** provide medical advice. Always consult a qualified healthcare provider for health concerns.

---

## 🚀 Demo
Try it live on Streamlit Cloud: [Healthcare FAQ Bot](https://healthbotfaq.streamlit.app/)

---

## ✨ Features
- Uploads a small healthcare FAQ knowledge base
- Embeds documents with OpenAI embeddings (`text-embedding-3-small`)
- Retrieves the most relevant FAQs using cosine similarity
- Provides concise, grounded answers with inline **citations [1], [2]**
- Built with **Streamlit** for a clean web app interface

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) → frontend & UI
- [OpenAI API](https://platform.openai.com/) → embeddings + GPT model
- [NumPy](https://numpy.org/) → vector math
- Python 3.9+

---

## 📦 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/healthcare-faq-bot.git
   cd healthcare-faq-bot
