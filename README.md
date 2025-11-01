# 🚀 ReadFusion – Multimodal RAG with Gemini

**ReadFusion** is an intelligent **Retrieval-Augmented Generation (RAG)** system powered by **Google Gemini** and **Streamlit**.  
It allows users to upload PDFs and interact with them using natural language — fusing **retrieval**, **reasoning**, and **multimodal understanding**.

---

## 🌟 Features
- 📄 PDF document upload & text extraction  
- 🧠 Context-aware answers using **RAG architecture**  
- 🎨 Built with **Streamlit** for an interactive UI  
- 🔍 Semantic search with **Sentence Transformers**  
- 🤖 Powered by **Gemini 2.5 Pro** for multimodal intelligence  

---

## 🧰 Tech Stack
- **Python 3.10+**
- **Streamlit**
- **Google Generative AI (Gemini)**
- **Sentence Transformers**
- **FAISS / vector embeddings**

---

## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/ranjithjunnu09/ReadFusion.git
cd ReadFusion

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # (Windows)
# source venv/bin/activate (Mac/Linux)

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo GOOGLE_API_KEY=your_key_here > .env
