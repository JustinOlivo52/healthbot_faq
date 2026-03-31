import os
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

# Read key from env or Streamlit Cloud Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Healthcare FAQ Bot", page_icon="🩺", layout="centered")
st.title("🩺 Healthcare FAQ Bot")
st.caption("Ask a health question and get a cited answer. Educational only — not medical advice.")

# -----------------------------
# Tiny knowledge base
# (add url if you want to show references)
# -----------------------------
FAQS: List[Dict] = [
    {
        "id": "faq1",
        "title": "Hydration Basics",
        "text": """How much water should I drink daily?
Most adults do well with steady hydration throughout the day.
Pale-yellow urine is a quick self-check. Increase fluids in heat or after long exercise."""
    },
    {
        "id": "faq2",
        "title": "Fever Guidance",
        "text": """When should I see a doctor for a fever?
Seek care if fever is very high, persists >3 days, or is paired with severe symptoms.
Stay hydrated and rest. Follow clinician instructions."""
    },
    {
        "id": "faq3",
        "title": "Sleep Hygiene",
        "text": """How can I sleep better?
Aim for 7–9 hours. Keep a consistent schedule.
Use a cool, dark room and avoid bright screens late at night."""
    },
    {
        "id": "faq4",
        "title": "Exercise for Beginners",
        "text": """How often should I work out?
Start with 2–3 sessions per week.
Focus on form and gradual progression.
Consult a clinician if you have medical concerns."""
    },
]

# -----------------------------
# Embeddings utils
# -----------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

@st.cache_resource(show_spinner=False)
def build_index(faqs: List[Dict]):
    texts = [f["text"] for f in faqs]
    vecs = embed_texts(texts)  # (N, d)
    return vecs

V = build_index(FAQS)  # vector matrix

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T

def retrieve(query: str, k: int = 2) -> List[Tuple[float, Dict]]:
    q = embed_texts([query])          # (1, d)
    sims = cosine_sim(q, V)[0]        # (N,)
    idx = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), FAQS[i]) for i in idx]

def build_context(hits: List[Tuple[float, Dict]]) -> str:
    lines = []
    for i, (_, f) in enumerate(hits, 1):
        lines.append(f"[{i}] {f['title']} ({f['id']})\n{f['text']}")
    return "\n---\n".join(lines)

SYSTEM = (
    "You are a helpful assistant. Use ONLY the provided context to answer. "
    "If not in the context, say you don't know. Cite sources inline as [1], [2]. "
    "This app is educational only and does not replace professional medical advice."
)

def answer(query: str, k: int = 2) -> str:
    hits = retrieve(query, k=k)
    context = build_context(hits)
    user_msg = (
        f"User question: {query}\n\n"
        f"Context (numbered sources):\n{context}\n\n"
        "Answer briefly (2–4 sentences) and include bracket citations."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content, hits

# -----------------------------
# UI
# -----------------------------
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Set it in your environment or Streamlit secrets.")
    st.stop()

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This app uses **RAG (Retrieval-Augmented Generation)** to answer basic healthcare questions. "
        "It searches a knowledge base of FAQs and uses GPT to write a cited answer grounded in those sources."
    )
    st.divider()
    k = st.slider("Number of sources", 1, 4, 2)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hi! I'm a healthcare FAQ assistant. Ask me about hydration, sleep, fever, or exercise and I'll answer with cited sources.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("hits"):
            with st.expander("Sources"):
                for i, (_, f) in enumerate(msg["hits"], 1):
                    st.markdown(f"**[{i}] {f['title']}** — {f['id']}")
                    st.write(f["text"])

if prompt := st.chat_input("Ask a health question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply, hits = answer(prompt, k=k)
        st.write(reply)
        with st.expander("Sources"):
            for i, (_, f) in enumerate(hits, 1):
                st.markdown(f"**[{i}] {f['title']}** — {f['id']}")
                st.write(f["text"])

    st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})
