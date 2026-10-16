import os
import sys
import time
import uuid
from pathlib import Path
import streamlit as st
import requests

# Add project root to sys.path so modules are discoverable on Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Inject Streamlit secrets into environment variables for configuration
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = v
except Exception:
    pass

# Read API base URL from secrets or environment, defaulting to local FastAPI
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Ensure corpus is indexed on Streamlit Cloud if running standalone
try:
    from app.services.vector_store import get_all_indexed_documents
    from app.ingestion.indexer import document_indexer
    if not get_all_indexed_documents():
        document_indexer.index_directory()
except Exception:
    pass

st.set_page_config(
    page_title="Technical Docs Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session ID for conversational memory
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:6]}"

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.sidebar.title("⚙️ Session Controls")
st.sidebar.info(f"**Session ID:** `{st.session_state['session_id']}`")
if st.sidebar.button("🔄 Clear Conversation Memory"):
    st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:6]}"
    st.session_state["chat_history"] = []
    st.session_state.pop("last_result", None)
    st.rerun()

st.title("🤖 Self-Corrective RAG Documentation Assistant")
st.caption("Powered by LangGraph (CRAG + Self-RAG), FastAPI, and ChromaDB — Express Analytics Assessment")

tab_chat, tab_docs, tab_ingest = st.tabs(["💬 Conversational Q&A", "📚 Indexed Corpus", "📥 Ingest New Docs"])

def execute_query(question: str, session_id: str):
    """
    Attempts to call the FastAPI backend. If the backend is unreachable
    (such as during standalone Streamlit Cloud deployment), executes the
    LangGraph pipeline in-process directly.
    """
    # 1. Try FastAPI REST backend
    if API_BASE_URL:
        try:
            res = requests.post(
                f"{API_BASE_URL}/query",
                json={"question": question, "session_id": session_id},
                timeout=30
            )
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

    # 2. Standalone In-Process Execution Fallback
    from app.graph.workflow import rag_app
    from app.services.memory_service import memory_service
    from app.core.config import settings

    start_time = time.perf_counter()
    history_str = memory_service.format_history_for_prompt(session_id)
    initial_state = {
        "question": question,
        "query_type": None,
        "transformed_query": question,
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": settings.MAX_RETRIES,
        "is_grounded": None,
        "web_fallback_used": False,
        "chat_history": history_str,
    }

    final_state = rag_app.invoke(initial_state)
    latency = round(time.perf_counter() - start_time, 3)
    answer = final_state.get("generation", "No answer generated.")
    memory_service.add_interaction(session_id, question, answer)

    citations = []
    for doc in final_state.get("relevant_docs", []):
        citations.append({
            "source": doc.get("source", "unknown"),
            "section": doc.get("section", ""),
            "snippet": doc.get("content", "")[:180] + "..." if len(doc.get("content", "")) > 180 else doc.get("content", "")
        })

    return {
        "question": question,
        "answer": answer,
        "sources": citations,
        "query_type": final_state.get("query_type"),
        "retries_used": final_state.get("retry_count", 0),
        "web_fallback_used": final_state.get("web_fallback_used", False),
        "is_grounded": final_state.get("is_grounded", True),
        "session_id": session_id,
        "execution_time_sec": latency
    }

with tab_chat:
    st.subheader("Ask a Question (Follow-ups Supported)")
    st.markdown("Ask anything about FastAPI dependencies, background tasks, security, or middleware.")

    # Display dialogue history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg:
                m = msg["metadata"]
                st.caption(f"Type: `{m.get('query_type')}` | Retries: `{m.get('retries_used')}` | Grounded: `{m.get('is_grounded')}`")

    # User query input
    user_query = st.chat_input("Enter your technical question or follow-up:")

    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        
        with st.spinner("Analyzing query, retrieving documents, and verifying relevance..."):
            try:
                data = execute_query(user_query, st.session_state["session_id"])
                st.session_state["last_result"] = data
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": data["answer"],
                    "metadata": {
                        "query_type": data.get("query_type"),
                        "retries_used": data.get("retries_used"),
                        "is_grounded": data.get("is_grounded"),
                        "sources": data.get("sources", [])
                    }
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error processing question: {e}")

    # Show details for latest assistant response
    if "last_result" in st.session_state:
        res_data = st.session_state["last_result"]
        
        st.divider()
        st.markdown("### 📊 Latest Response Telemetry & Citations")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Query Intent", res_data.get("query_type") or "N/A")
        with col2:
            st.metric("Rewrite Retries", res_data.get("retries_used", 0))
        with col3:
            st.metric("Latency", f"{res_data.get('execution_time_sec', 0)}s")
        with col4:
            grounded_label = "✅ Grounded" if res_data.get("is_grounded", True) else "⚠️ Hallucination Warning"
            st.metric("Self-RAG Grounded", grounded_label)

        if res_data.get("sources"):
            st.markdown("#### 📖 Citations & Sources Grounding")
            for i, src in enumerate(res_data["sources"]):
                with st.expander(f"Source {i+1}: {src.get('source')} (Section: {src.get('section') or 'General'})"):
                    st.markdown(f"**Snippet Preview:**\n> {src.get('snippet')}")

        st.divider()
        st.markdown("#### Was this response helpful?")
        fb_col1, fb_col2 = st.columns([1, 4])
        with fb_col1:
            rating = st.radio("Rating", ["👍 Up", "👎 Down"], horizontal=True, key="feedback_radio")
        with fb_col2:
            comment = st.text_input("Optional feedback comments:", key="feedback_comment")
        
        if st.button("Submit Feedback"):
            rating_val = "up" if "Up" in rating else "down"
            try:
                from app.services.feedback_service import feedback_service
                feedback_service.record_feedback(
                    query=res_data["question"],
                    answer=res_data["answer"],
                    rating=rating_val,
                    comment=comment
                )
                st.success("Feedback submitted successfully!")
            except Exception:
                st.error("Failed to submit feedback.")

with tab_docs:
    st.subheader("Currently Indexed Technical Documents")
    if st.button("Refresh Corpus View"):
        pass
    try:
        from app.services.vector_store import get_all_indexed_documents
        docs_data = get_all_indexed_documents()
        total_chunks = sum(d["chunks_count"] for d in docs_data)
        st.info(f"Total Sources: {len(docs_data)} | Total Indexed Chunks: {total_chunks}")
        for doc in docs_data:
            with st.expander(f"📄 {doc['source']} — {doc['chunks_count']} chunk(s)"):
                st.write("Indexed Sections:")
                for sec in doc.get("sections", []):
                    st.markdown(f"- {sec}")
    except Exception as e:
        st.error(f"Cannot load document index: {e}")

with tab_ingest:
    st.subheader("Add New Documentation to Corpus")
    st.markdown("Upload markdown files or supply documentation URLs to ingest into the vector store.")

    st.markdown("#### Option 1: File Upload")
    uploaded_files = st.file_uploader("Upload Markdown/Text files", accept_multiple_files=True, type=["md", "txt"])
    if st.button("Upload and Index Files"):
        if uploaded_files:
            try:
                from app.ingestion.indexer import document_indexer
                added = 0
                for f in uploaded_files:
                    chunks = document_indexer.chunker.chunk_markdown_text(
                        f.getvalue().decode("utf-8"),
                        source_name=f.name
                    )
                    if chunks:
                        document_indexer.vector_store.add_documents(chunks)
                    added += len(chunks)
                st.success(f"Indexed successfully! Added {added} chunks.")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.markdown("#### Option 2: Ingest from URL")
    url_input = st.text_input("Documentation raw URL (e.g. GitHub raw markdown URL):")
    if st.button("Ingest from URL"):
        if url_input:
            try:
                from app.ingestion.indexer import document_indexer
                count = document_indexer.index_url(url_input)
                st.success(f"Indexed URL! Added {count} chunks.")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
