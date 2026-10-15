# 🎬 Express Analytics Assessment: Live Demo & Interview Guide

This guide provides a structured, step-by-step walkthrough to present the **Self-Corrective RAG Technical Documentation Assistant** during review calls or live technical interviews.

---

## 🎯 1. 2-Minute Elevator Pitch
> *"Traditional RAG applications suffer from two critical flaws: noisy, irrelevant context polluting the generation prompt, and static single-pass retrieval that fails when a user query is ambiguous. In this project, I implemented a **Self-Corrective RAG (CRAG)** architecture orchestrated with **LangGraph**. The pipeline dynamically analyzes intent, retrieves code-aware chunks, grades chunk relevance with an LLM evaluator, adaptively rewrites queries with guaranteed loop termination, validates answers for hallucinations via Self-RAG, and provides source citations. It is fully served via a production **FastAPI** backend and an interactive **Streamlit** frontend with session conversational memory."*

---

## 🕹️ 2. Step-by-Step Live Demo Script

### Preparation
1. Ensure both services are running:
   - FastAPI Backend: `http://localhost:8000`
   - Streamlit Frontend: `http://localhost:8501`
2. Open the Streamlit UI in your browser at `http://localhost:8501`.

---

### Step 1: In-Domain Grounded Q&A with Inline Citations
- **Action:** In the Q&A input box, enter:
  > *"How do I use yield in dependencies to clean up database connections in FastAPI?"*
- **What to highlight:**
  1. **Intent Classification:** Point out the telemetry badge showing `Query Intent: how-to`.
  2. **Grounded Answer:** The answer synthesizes code with `async def get_db(): ... yield db ... finally: db.close()`.
  3. **Citations:** Click the **📖 Citations & Sources Grounding** expander. Show that the exact document (`fastapi_dependency_injection.md`) and section hierarchy (`FastAPI Dependencies > Dependencies with yield (Cleanup & DB Sessions)`) are cited.
  4. **Self-RAG Grounded:** Point out the `Self-RAG Grounded: ✅ Grounded` metric proving zero hallucination.

---

### Step 2: Conversational Memory & Coreference Resolution (Bonus)
- **Action:** Ask a follow-up question in the same chat without repeating the subject:
  > *"Can I also inject background tasks inside them?"*
- **What to highlight:**
  1. The user said *"inside them"* without specifying dependencies.
  2. The `SessionMemoryService` passed the prior turn context to `query_analysis_node`.
  3. The system correctly identifies that *"them"* refers to FastAPI dependencies and retrieves `Background Tasks in Dependencies` from `fastapi_background_tasks.md`.

---

### Step 3: Self-Correction Loop on Ambiguous / Out-of-Domain Queries
- **Action:** Enter a completely out-of-domain prompt:
  > *"How do I bake sourdough bread with active dry yeast?"*
- **What to highlight:**
  1. Vector search retrieves chunks from the database, but the **Document Grader correctly rejects all of them as irrelevant** (`relevant_docs = 0`).
  2. **Self-Correction Loop:** The system rewrites the query and tries again.
  3. **Loop Termination:** Point out `Rewrite Retries: 2 / 2`. To prevent infinite loops, the state machine strictly halts at `max_retries`.
  4. **Graceful Fallback:** Routes to DuckDuckGo web search fallback or gracefully states context is insufficient, instead of hallucinating recipe steps.

---

### Step 4: Feedback Collection & Corpus Ingestion
- **Action 1:** On any generated answer, click the **👍 Up** radio button, add a short comment (*"Clear and cited"*), and click **Submit Feedback**.
  - Show `data/feedback.json` updating in real time.
- **Action 2:** Switch to the **📚 Indexed Corpus** tab to show all 4 documentation files and their indexed sections.
- **Action 3:** Switch to the **📥 Ingest New Docs** tab and upload a sample markdown file or input a URL to show real-time indexing without server restarts.

---

## 💡 3. Anticipated Technical Questions & Strong Answers

### Q1: Why did you choose LangGraph over LangChain's standard LCEL chains?
> **Answer:** *"Standard LCEL chains are Directed Acyclic Graphs (DAGs)—they only execute linearly forward. A true self-corrective system requires cyclical execution: retrieve $\rightarrow$ grade $\rightarrow$ if poor, rewrite query $\rightarrow$ loop back to retrieve. LangGraph provides first-class support for cycles, state management, and conditional edge routing, allowing us to implement the CRAG (Corrective RAG) pattern cleanly."*

### Q2: How did you prevent the query rewriting loop from executing infinitely?
> **Answer:** *"In the LangGraph `GraphState` TypedDict, we maintain an explicit integer counter `retry_count` alongside a ceiling `max_retries` (default 2). In the conditional routing function `route_after_grading`, if no relevant chunks are found but `retry_count >= max_retries`, the state machine forces a transition to the web fallback node rather than cycling back to query rewriting."*

### Q3: Why did you implement a two-stage chunking strategy instead of naive character chunking?
> **Answer:** *"Technical documentation contains code blocks and nested markdown headers. A naive character chunker will split code snippets right down the middle, creating syntax errors and destroying function context. Our two-stage pipeline first splits along `#`, `##`, and `###` headers using `MarkdownHeaderTextSplitter`, injecting the section path into chunk metadata. Then, `RecursiveCharacterTextSplitter` respects markdown code fences (` ``` `), ensuring code examples remain intact."*

### Q4: How is hallucination prevented?
> **Answer:** *"We use two complementary layers: First, in the generation prompt, the LLM is restricted to strictly synthesize using the provided context blocks and must cite sources. Second, we added a Self-RAG inspired `hallucination_check_node` that takes the generated output and evaluates factual consistency against the retrieved text, flagging ungrounded answers."*

### Q5: What would you do differently with a larger corpus in production?
> **Answer:** *"For production scale, I would:
> 1. Implement **Hybrid Search (Dense + Sparse BM25)** using reciprocal rank fusion (RRF) for better exact match retrieval of class names and flags.
> 2. Introduce a **Reranker (e.g. Cohere Rerank or BGE-Reranker)** to re-score the top 20 candidates down to top 4.
> 3. Implement **Parent-Child Chunking**: index small semantic sentence windows for vector retrieval but pass the parent section to the LLM for generation context."*
