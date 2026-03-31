# 🩺 Healthcare FAQ Bot (RAG Demo)

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** and **OpenAI**.
Ask health questions about hydration, fever, sleep, or exercise and get concise, cited answers grounded in a curated FAQ knowledge base.

⚠️ **Disclaimer:** This app is for educational purposes only and does **not** provide medical advice. Always consult a qualified healthcare provider for health concerns.

---

## 🚀 Demo
Try it live on Streamlit Cloud: [Healthcare FAQ Bot](https://healthbotfaq.streamlit.app/)

---

## ✨ Features
- Multi-turn conversation history — ask follow-up questions naturally
- Embeds a healthcare FAQ knowledge base using OpenAI embeddings (`text-embedding-3-small`)
- Retrieves the most relevant FAQs using cosine similarity
- Generates concise, grounded answers with inline **citations [1], [2]**
- Sources expander on every answer so you can verify what the bot used
- Built with **Streamlit** for a clean, chat-style web interface

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) → frontend & chat UI
- [OpenAI API](https://platform.openai.com/) → embeddings (`text-embedding-3-small`) + chat (`gpt-4o-mini`)
- [NumPy](https://numpy.org/) → vector math / cosine similarity
- Python 3.9+

---

## 📦 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/JustinOlivo52/healthbot_faq.git
   cd healthbot_faq
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🔑 Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. In **Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy — the app will be live at your Streamlit Cloud URL
