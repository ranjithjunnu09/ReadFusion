import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from utils.text_extraction import extract_text_from_pdf, extract_text_from_docx
from utils.vector_store import add_to_vector_store, query_vector_store
from PIL import Image

# Load env vars
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-pro")

st.set_page_config(page_title="🧠 Multimodal RAG", layout="wide")
st.title("🧠 ReadFusion ")

uploaded_file = st.file_uploader("📤 Upload a file (PDF, DOCX, or Image)", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    text_data = ""
    image_data = None

    if uploaded_file.type == "application/pdf":
        text_data = extract_text_from_pdf(uploaded_file)
        st.text_area("📄 Extracted Text:", text_data[:1000], height=200)
        add_to_vector_store(text_data.split("\n"))

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text_data = extract_text_from_docx(uploaded_file)
        st.text_area("📄 Extracted Text:", text_data[:1000], height=200)
        add_to_vector_store(text_data.split("\n"))

    elif "image" in uploaded_file.type:
        image_data = Image.open(uploaded_file)
        st.image(image_data, caption="🖼 Uploaded Image", use_column_width=True)

    query = st.text_input("💬 Ask a question about your uploaded content:")

    if st.button("🚀 Generate Answer"):
        context = ""
        if text_data:
            context_chunks = query_vector_store(query)
            context = "\n".join(context_chunks)

        if image_data:
            response = model.generate_content([query, image_data])
        else:
            response = model.generate_content(f"Context:\n{context}\n\nQuestion:\n{query}")

        st.markdown("### 🧠 Answer:")
        st.write(response.text)
