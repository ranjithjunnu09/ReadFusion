# app_improved.py
import os
import io
import time
from typing import List, Tuple
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss

# ------------- CONFIG -------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your .env")

genai.configure(api_key=API_KEY)

# Choose a default model (change if you prefer)
DEFAULT_MODEL = "models/gemini-2.5-pro"

# ------------- STYLES -------------
NEON_CSS = """
<style>
/* page background */
body { background: #071226; color: #E6F1FF; }
/* card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 10px;
  padding: 16px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.6);
  margin-bottom: 12px;
}
.stButton>button { background: linear-gradient(90deg,#34E89E,#2575FC); color: white; border: none; }
textarea { color: #E6F1FF; background: #0b1220; }
code { color: #34E89E; }
h1 { color: #EAF6FF; }
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ------------- RAG (Simple FAISS wrapper) -------------
class SimpleRAG:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2"):
        self.embed = SentenceTransformer(embed_model_name)
        self.index = None
        self.metadatas = []  # each -> {"text":..., "source":..., "type":"text"|"image"}
        self.d = None

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        tokens = text.split()
        chunks = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i:i+chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap
        return chunks

    def add_text(self, text: str, source: str, chunk_size: int, overlap: int):
        chunks = self.chunk_text(text, chunk_size, overlap)
        items = [{"text":c, "source": source, "type": "text"} for c in chunks]
        self._index_items(items)

    def add_image_caption(self, caption: str, source: str):
        # store small caption / alt-text for image
        items = [{"text": caption, "source": source, "type": "image"}]
        self._index_items(items)

    def _index_items(self, items: List[dict]):
        docs = [it["text"] for it in items]
        import numpy as np
        embs = self.embed.encode(docs, convert_to_numpy=True)
        faiss.normalize_L2(embs)
        if self.index is None:
            self.d = embs.shape[1]
            self.index = faiss.IndexFlatIP(self.d)
            self.index.add(embs)
            self.metadatas = items.copy()
        else:
            self.index.add(embs)
            self.metadatas.extend(items.copy())

    def query(self, q: str, k: int = 4) -> List[Tuple[dict, float]]:
        if self.index is None:
            return []
        import numpy as np
        q_emb = self.embed.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        res = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.metadatas):
                res.append((self.metadatas[idx], float(score)))
        return res

# ------------- Helpers -------------
def extract_text_from_pdf_bytes(b: bytes) -> str:
    reader = PdfReader(io.BytesIO(b))
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def extract_images_from_pdf_bytes(b: bytes) -> List[Image.Image]:
    imgs = []
    reader = PdfReader(io.BytesIO(b))
    for p in reader.pages:
        # p.images is new in pypdf; fall back if not available
        try:
            for img in p.images:
                try:
                    pil = Image.open(io.BytesIO(img.data))
                    imgs.append(pil.convert("RGB"))
                except Exception:
                    pass
        except Exception:
            pass
    return imgs

def save_text_as_docx_buf(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

def save_text_as_pdf_buf(text: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=letter)
    width, height = letter
    margin = 72
    y = height - margin
    for paragraph in text.split("\n"):
        while len(paragraph) > 100:
            part = paragraph[:100]
            c.drawString(margin, y, part)
            paragraph = paragraph[100:]
            y -= 14
            if y < margin:
                c.showPage(); y = height - margin
        if paragraph:
            c.drawString(margin, y, paragraph)
            y -= 14
        if y < margin:
            c.showPage(); y = height - margin
    c.save()
    bio.seek(0)
    return bio.read()

def generate_with_gemini(prompt: str, model_name: str, max_tokens: int = 512) -> str:
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        return f"ERROR: {e}"

# ------------- Session init -------------
if "rag" not in st.session_state:
    st.session_state["rag"] = SimpleRAG()
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of {"query":.., "answer":.., "retrieved": [...]}

rag: SimpleRAG = st.session_state["rag"]

# ------------- UI -------------
st.title("ReadFusion — Multimodal RAG")
st.markdown("**Upload PDFs / DOCX / Images** — the system will extract text & images, index them, and let you ask questions grounded in the uploaded content.")

# Left column: upload + settings
left, right = st.columns([1, 2])
with left:
    st.markdown("<div class='card'><strong>1. Upload files</strong></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF / DOCX / PNG / JPG / TXT (multiple)", accept_multiple_files=True, type=["pdf","docx","png","jpg","jpeg","txt"])
    st.markdown("<div class='card'><strong>2. Index settings</strong></div>", unsafe_allow_html=True)
    chunk_size = st.number_input("Chunk size (words)", min_value=100, max_value=2000, value=400, step=50)
    chunk_overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=400, value=60, step=10)
    top_k = st.number_input("Top-K retrieved", min_value=1, max_value=10, value=4, step=1)
    model_choice = st.selectbox("Model", options=[DEFAULT_MODEL, "models/gemini-2.5-flash","models/gemini-2.5-pro-preview-03-25"], index=0)
    if st.button("Reset index"):
        st.session_state.pop("rag", None)
        st.session_state.pop("history", None)
        st.experimental_rerun()

    if uploaded:
        total = len(uploaded)
        progress = st.progress(0)
        i = 0
        for f in uploaded:
            i += 1
            name = f.name
            b = f.read()
            st.write(f"Processing: **{name}**")
            # text
            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf_bytes(b)
                rag.add_text(text, source=name, chunk_size=chunk_size, overlap=chunk_overlap)
                # extract images
                imgs = extract_images_from_pdf_bytes(b)
                for idx, im in enumerate(imgs):
                    caption = f"[Image extracted from {name} - image {idx+1}]"
                    rag.add_image_caption(caption, source=name)
                st.write(f" - indexed text chunks: approx {max(1, len(text.split())//chunk_size)}")
                st.write(f" - extracted images: {len(imgs)}")
            elif name.lower().endswith(".docx"):
                # docx
                try:
                    doc = Document(io.BytesIO(b))
                    txt = "\n".join(p.text for p in doc.paragraphs)
                    rag.add_text(txt, source=name, chunk_size=chunk_size, overlap=chunk_overlap)
                    st.write(f" - indexed docx chunks")
                except Exception as e:
                    st.write("docx parse error:", e)
            elif any(name.lower().endswith(ext) for ext in ["png","jpg","jpeg"]):
                # small note: we store simple caption that image exists; real image embedding would require visual embedder
                rag.add_image_caption(f"[User image: {name}]", source=name)
                st.image(Image.open(io.BytesIO(b)), caption=f"Uploaded: {name}", use_column_width=True)
            elif name.lower().endswith(".txt"):
                try:
                    txt = b.decode("utf-8")
                except Exception:
                    txt = b.decode("latin-1")
                rag.add_text(txt, source=name, chunk_size=chunk_size, overlap=chunk_overlap)
                st.write(" - indexed text")
            progress.progress(i/total)
        st.success("Ingestion complete.")

with right:
    st.markdown("<div class='card'><strong>Ask a question</strong></div>", unsafe_allow_html=True)
    query = st.text_area("Enter your question here", height=140)
    if st.button("Generate Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            if rag.index is None:
                st.warning("Index is empty — upload documents first.")
            else:
                with st.spinner("Retrieving and generating..."):
                    hits = rag.query(query, k=top_k)
                    # show retrieved
                    st.markdown("**Retrieved snippets:**")
                    retrieved_str = ""
                    for idx, (meta, score) in enumerate(hits):
                        st.markdown(f"**{idx+1}.** Source: `{meta['source']}` • score: `{score:.3f}`")
                        st.write(meta["text"][:800] + ("..." if len(meta["text"])>800 else ""))
                        retrieved_str += meta["text"] + "\n\n"
                    # build prompt with context
                    prompt = (
                        "You are an assistant that must answer using ONLY the provided context. "
                        "If the answer is not in the context, say 'I don't know'.\n\n"
                        f"Context:\n{retrieved_str}\n\nQuestion: {query}\nAnswer:"
                    )
                    answer = generate_with_gemini(prompt, model_name=model_choice)
                    # save to history
                    st.session_state["history"] = st.session_state.get("history", [])
                    st.session_state["history"].insert(0, {"query": query, "answer": answer, "retrieved": hits})
                    # show answer card
                    st.markdown("<div class='card'><strong>Answer</strong></div>", unsafe_allow_html=True)
                    st.write(answer)
                    # download buttons
                    pdf_bytes = save_text_as_pdf_buf(answer)
                    docx_bytes = save_text_as_docx_buf(answer)
                    c1, c2 = st.columns(2)
                    c1.download_button("Download PDF", data=pdf_bytes, file_name="readfusion_answer.pdf", mime="application/pdf")
                    c2.download_button("Download DOCX", data=docx_bytes, file_name="readfusion_answer.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# Sidebar: history & index stats
with st.sidebar:
    st.header("Conversation History")
    if st.session_state.get("history"):
        for i, item in enumerate(st.session_state["history"][:10]):
            st.markdown(f"**Q{i+1}:** {item['query']}")
            with st.expander("Show answer"):
                st.write(item["answer"])
    else:
        st.write("No history yet.")

    st.markdown("---")
    st.write("Indexed chunks:", len(rag.metadatas) if rag.metadatas else 0)
    st.caption("ReadFusion — Where Knowledge Meets Intelligence")
