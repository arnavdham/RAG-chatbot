import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
import openai
import os
from sentence_transformers import SentenceTransformer
from timeit import default_timer as timer
# import spacy
from spacy.lang.en import English

# Math rendering CSS
st.markdown("""
<style>
.katex-html {
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# Set environment variables
os.environ["OPENROUTER_API_KEY"] = "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"

# Mean pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to call OpenRouter model
def call_openrouter_llm(context, question, model):
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    prompt = f"""Use this physics textbook context to answer accurately. Format all math expressions in LaTeX using double dollar signs ($$ ... $$):

Context:
{context}

Question:
{question}

Answer:"""

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://physics-rag-app.com",
            "X-Title": "RAG Pipeline for PDF Parsing",
        },
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "fine_tuned_minilm"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model

# Process PDF into chunks
# import spacy
# from PyPDF2 import PdfReader

def process_pdf(uploaded_file, chunk_target=751, tokenizer=None):
    # Initialize spaCy English pipeline with sentencizer
    nlp = English()
    nlp.add_pipe("sentencizer")
    
    reader = PdfReader(uploaded_file)
    pages_and_chunks = []
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            continue

        # Use spaCy to split into sentences
        doc = nlp(page_text)
        sentences = [str(sent).strip() for sent in doc.sents if str(sent).strip()]

        # Chunk sentences to target size
        current_chunk = []
        char_count = 0
        for sent in sentences:
            if char_count + len(sent) > chunk_target and current_chunk:
                chunk_text = " ".join(current_chunk)
                pages_and_chunks.append({
                    "page_number": page_num + 1,
                    "sentence_chunk": chunk_text,
                    "chunk_char_count": len(chunk_text),
                    "chunk_word_count": len(chunk_text.split()),
                    "chunk_token_count": len(tokenizer.encode(chunk_text, truncation=True)) if tokenizer else None
                })
                current_chunk = []
                char_count = 0
            current_chunk.append(sent)
            char_count += len(sent)
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            pages_and_chunks.append({
                "page_number": page_num + 1,
                "sentence_chunk": chunk_text,
                "chunk_char_count": len(chunk_text),
                "chunk_word_count": len(chunk_text.split()),
                "chunk_token_count": len(tokenizer.encode(chunk_text, truncation=True)) if tokenizer else None
            })
    return pages_and_chunks



# Compute embeddings
def compute_embeddings(pages_and_chunks, model, tokenizer):
    embeddings = []
    for chunk in pages_and_chunks:
        inputs = tokenizer(chunk["sentence_chunk"], return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
        embeddings.append(embedding[0])
    return embeddings

# Create FAISS index
def create_faiss_index(embeddings, dimension=384):
    index = faiss.IndexFlatIP(dimension)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# Retrieve relevant resources using FAISS
def retrieve_relevant_resources_faiss(query: str, model: SentenceTransformer, faiss_index, n_resources_to_return: int = 5, print_time: bool = True):
    query_embedding = model.encode(query, convert_to_tensor=False).astype("float32")
    query_embedding = np.expand_dims(query_embedding, axis=0)  # Expand dimensions to match FAISS format
    
    # Perform FAISS search
    start_time = timer()
    distances, indices = faiss_index.search(query_embedding, n_resources_to_return)
    end_time = timer()
    
    if print_time:
        print(f"[FAISS] Search time: {end_time - start_time:.4f}s")
    
    return distances[0], indices[0]

# Get top-k context from FAISS
def get_top_k_context_faiss(query, model, faiss_index, pages_and_chunks, k=5):
    _, top_indices = retrieve_relevant_resources_faiss(query, model, faiss_index, n_resources_to_return=k)
    context = "\n\n".join(pages_and_chunks[i]["sentence_chunk"] for i in top_indices)
    return context, top_indices

# Wrapper for querying with FAISS RAG
def ask_question_with_faiss_rag(query, model, faiss_index, pages_and_chunks, k=5, model_choice="meta-llama/llama-3.1-8b-instruct"):
    context, _ = get_top_k_context_faiss(query, model, faiss_index, pages_and_chunks, k)
    answer = call_openrouter_llm(context, query, model_choice)
    return answer, context

# UI and processing
if "processed" not in st.session_state:
    st.session_state.processed = False

st.title("RAG Pipeline Interface")
st.subheader("with OpenRouter Model Selection")

with st.sidebar:
    st.header("Model Settings")
    model_choice = st.radio("Choose LLM", ["deepseek/deepseek-r1", "google/gemma-3-27b-it", "meta-llama/llama-3.1-8b-instruct"], index=0)
    top_k = st.slider("Results to Show", 1, 5, 3)
    chunk_target = st.number_input("Chunk Size Target", 500, 1000, 751)

tokenizer, model = load_model()
embedding_model = SentenceTransformer("fine_tuned_minilm")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file and not st.session_state.processed:
    with st.spinner(f"Processing {uploaded_file.name}..."):
        pages_and_chunks = process_pdf(uploaded_file, chunk_target)
        embeddings = compute_embeddings(pages_and_chunks, model, tokenizer)
        index = create_faiss_index(embeddings)
        st.session_state.update({"processed": True, "index": index, "pages_and_chunks": pages_and_chunks})
        st.success(f"Processed {len(pages_and_chunks)} chunks (avg {chunk_target} chars)")

query = st.text_input("Enter question:")
if query and st.session_state.processed:
    with st.spinner("Searching..."):
        answer, context = ask_question_with_faiss_rag(
            query=query,
            model=embedding_model,
            faiss_index=st.session_state.index,
            pages_and_chunks=st.session_state.pages_and_chunks,
            k=top_k,
            model_choice=model_choice
        )

        st.subheader(f"Top {top_k} Results")
        top_chunks = context.split("\n\n")
        for chunk in top_chunks:
            for page in st.session_state.pages_and_chunks:
                if page["sentence_chunk"] == chunk:
                    with st.expander(f"Page {page['page_number']}"):
                        st.write(chunk)
                    break

        st.subheader("Generated Answer")
        st.markdown(answer, unsafe_allow_html=False)
