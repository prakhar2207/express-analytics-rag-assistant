import streamlit as st
import requests
import json

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Technical Docs Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Self-Corrective RAG Documentation Assistant")
st.caption("Powered by LangGraph, FastAPI, and ChromaDB — Built for Express Analytics Assessment")

tab_chat, tab_docs, tab_ingest = st.tabs(["💬 Assistant Q&A", "📚 Indexed Corpus", "📥 Ingest New Docs"])

with tab_chat:
    st.subheader("Ask a Question")
    st.markdown("Ask anything about FastAPI dependencies, background tasks, security, or middleware.")

    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g. How do I use dependencies with yield to clean up database connections?"
    )

    if st.button("Submit Question", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing query, retrieving documents, and verifying relevance..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/query", json={"question": user_query}, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state["last_query"] = user_query
                        st.session_state["last_result"] = data
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to FastAPI backend: {e}. Ensure `uvicorn app.main:app` is running.")

    if "last_result" in st.session_state:
        res_data = st.session_state["last_result"]
        st.markdown("### Answer")
        st.markdown(res_data["answer"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Query Type", res_data.get("query_type") or "N/A")
        with col2:
            st.metric("Rewrites / Retries", res_data.get("retries_used", 0))
        with col3:
            st.metric("Latency", f"{res_data.get('execution_time_sec', 0)}s")

        if res_data.get("sources"):
            st.markdown("### 📖 Citations & Sources Grounding")
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
            fb_res = requests.post(
                f"{API_BASE_URL}/feedback",
                json={
                    "query": st.session_state["last_query"],
                    "answer": res_data["answer"],
                    "rating": rating_val,
                    "comment": comment
                }
            )
            if fb_res.status_code == 200:
                st.success("Feedback submitted successfully!")
            else:
                st.error("Failed to submit feedback.")

with tab_docs:
    st.subheader("Currently Indexed Technical Documents")
    if st.button("Refresh Corpus View"):
        pass
    try:
        r = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if r.status_code == 200:
            docs_data = r.json()
            st.info(f"Total Sources: {docs_data['total_documents']} | Total Indexed Chunks: {docs_data['total_chunks']}")
            for doc in docs_data["documents"]:
                with st.expander(f"📄 {doc['source']} — {doc['chunks_count']} chunk(s)"):
                    st.write("Indexed Sections:")
                    for sec in doc.get("sections", []):
                        st.markdown(f"- {sec}")
        else:
            st.error("Could not fetch document index.")
    except Exception as e:
        st.error(f"Cannot connect to backend: {e}")

with tab_ingest:
    st.subheader("Add New Documentation to Corpus")
    st.markdown("Upload markdown files or supply documentation URLs to ingest into the vector store.")

    st.markdown("#### Option 1: File Upload")
    uploaded_files = st.file_uploader("Upload Markdown/Text files", accept_multiple_files=True, type=["md", "txt"])
    if st.button("Upload and Index Files"):
        if uploaded_files:
            files_payload = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
            resp = requests.post(f"{API_BASE_URL}/ingest", files=files_payload)
            if resp.status_code == 200:
                st.success(f"Indexed successfully! Added {resp.json().get('total_chunks_added')} chunks.")
            else:
                st.error(f"Ingest failed: {resp.text}")

    st.markdown("#### Option 2: Ingest from URL")
    url_input = st.text_input("Documentation raw URL (e.g. GitHub raw markdown URL):")
    if st.button("Ingest from URL"):
        if url_input:
            resp = requests.post(f"{API_BASE_URL}/ingest", json={"urls": [url_input]})
            if resp.status_code == 200:
                st.success(f"Indexed URL! Added {resp.json().get('total_chunks_added')} chunks.")
            else:
                st.error(f"Ingest failed: {resp.text}")
