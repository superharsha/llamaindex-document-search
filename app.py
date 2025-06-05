import streamlit as st
import asyncio
import os

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# Function to get secrets from either Streamlit secrets or environment variables
def get_secret(key):
    try:
        # Try Streamlit secrets first (for deployment)
        return st.secrets[key]
    except (KeyError, AttributeError):
        # Fall back to environment variables (for local development)
        return os.getenv(key)

# Set Gemini API key
gemini_api_key = get_secret("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GEMINI_API_KEY"] = gemini_api_key

# ────────────────────────────────────────────────────────────────────────────────
# 1. Build (and cache) a local, BM25-only query engine with HyDE + Gemini
# ────────────────────────────────────────────────────────────────────────────────
# Dependencies: python-pptx, Pillow, openpyxl, docx2txt (lightweight, no torch/transformers)

@st.cache_resource
def build_local_query_engine(sim_top_k: int = 5):
    """
    1) Read all files under ./my_local_docs/
    2) Chunk them hierarchically (4096, 2048, 1024, 512 tokens)
    3) Populate an in-memory SimpleDocumentStore
    4) Build BM25Retriever → HyDE → Gemini
    """
    # 1) HyDE transformer (to rewrite queries + include original)
    hyde = HyDEQueryTransform(include_original=True)

    # 2) Read & chunk documents
    documents = SimpleDirectoryReader("./my_local_docs/").load_data()
    if not documents:
        raise RuntimeError("No files found in ./my_local_docs/. Please add some .txt/.pdf/.docx files.")

    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=(4096, 2048, 1024, 512))
    nodes = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    # 3) Build in-memory BM25 index
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=sim_top_k,
    )

    # 4) Attach Gemini as the final answer generator
    llm = GeminiMultiModal(model_name="gemini-2.0-flash", temperature=0)
    Settings.llm = llm

    # 5) Build a RetrieverQueryEngine (BM25 → simple synthesizer)
    primitive_engine = RetrieverQueryEngine.from_args(
        bm25_retriever,
        response_synthesizer=get_response_synthesizer(
            text_qa_template=PromptTemplate(
                """You are an expert document analyst. Use the provided context to answer the user's question accurately and comprehensively.

CONTEXT INFORMATION:
{context_str}

INSTRUCTIONS:
- Answer based ONLY on the information provided in the context above
- If the context doesn't contain enough information to fully answer the question, clearly state what information is missing
- Provide specific details, quotes, and examples from the context when relevant
- Structure your response clearly with proper formatting
- If multiple documents are referenced, distinguish between them in your answer
- If you find contradictory information, acknowledge and explain the discrepancies

QUESTION: {query_str}

ANSWER:"""
            ),
            response_mode="compact",
            use_async=True,
        ),
    )

    # 6) Wrap with HyDE → Gemini
    engine = TransformQueryEngine(primitive_engine, hyde)
    return engine

# Helper to call the async query from synchronous Streamlit code
def run_query(engine, query: str):
    return asyncio.run(engine.aquery(query))


# ────────────────────────────────────────────────────────────────────────────────
# 2. Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Local BM25 + HyDE Query", layout="wide")
st.title("📚 Local Document Search (BM25 + HyDE → Gemini)")

st.markdown("""
This app:
1. Reads **`./my_local_docs/`** (supports `.txt`, `.pdf`, `.docx`)  
2. Chunks everything into a BM25 index  
3. Wraps BM25 with HyDE → Gemini for answer generation  
4. Let's you type a query and see:
   - **Answer** (generated by Gemini)  
   - **Top matching chunks** (with BM25 scores)
""")

query = st.text_input("Enter your query here:", placeholder="e.g. What are the main points in document X?")

if st.button("🔍 Run Query") and query.strip():
    try:
        with st.spinner("🔄 Building index (if first run) and querying…"):
            engine = build_local_query_engine(sim_top_k=20)
            resp = run_query(engine, query)
    except Exception as e:
        st.error(f"Error: {e}")
    else:
        st.subheader("🗒️ Answer")
        st.write(resp.response)

        st.subheader("🔝 Top Chunks")
        for node_info in resp.source_nodes:
            txt = node_info.node.text.replace("\n", " ").strip()
            score = round(node_info.score, 3)
            # Show the first 200 characters of each chunk
            display_text = txt[:200] + ("…" if len(txt) > 200 else "")
            st.write(f"- {display_text}  _(score: {score})_")
