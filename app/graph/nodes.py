import os
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.services.llm_factory import get_llm
from app.services.vector_store import similarity_search_with_metadata
from app.core.config import settings

# --- Pydantic Data Structures for Structured Outputs ---

class QueryAnalysisOutput(BaseModel):
    query_type: str = Field(description="Classification: 'conceptual', 'how-to', 'troubleshooting', or 'api_reference'")
    expanded_query: str = Field(description="Rewritten search query with synonyms and clear technical keywords")

class DocumentGrade(BaseModel):
    is_relevant: bool = Field(description="True if the document chunk contains information relevant to answering the query, False otherwise")
    rationale: str = Field(description="Brief reason for the decision")

# --- Node Implementations ---

def query_analysis_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 1: Analyzes the user question, classifies intent, and optimizes query for retrieval.
    """
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert technical documentation assistant. "
            "Analyze the user's question about Python/FastAPI/frameworks.\n"
            "1. Classify query_type into one of: 'conceptual', 'how-to', 'troubleshooting', 'api_reference'.\n"
            "2. Generate an expanded_query that clarifies ambiguities and adds technical keywords or synonyms to optimize vector retrieval.\n"
            "Respond strictly in JSON format matching the schema."
        )),
        ("human", "Question: {question}")
    ])
    
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(QueryAnalysisOutput)
        chain = prompt | structured_llm
        analysis: QueryAnalysisOutput = chain.invoke({"question": question})
        return {
            "query_type": analysis.query_type,
            "transformed_query": analysis.expanded_query
        }
    except Exception:
        # Fallback in case structured output or API call has an issue
        return {
            "query_type": "how-to",
            "transformed_query": question
        }

def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 2: Retrieves top-k relevant document chunks from ChromaDB.
    """
    query = state.get("transformed_query") or state["question"]
    docs = similarity_search_with_metadata(query, k=settings.TOP_K)
    return {
        "documents": docs
    }

def document_grading_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 3: Evaluates retrieved chunks. LLM grades each chunk as relevant or irrelevant.
    """
    question = state["question"]
    docs = state.get("documents", [])
    
    if not docs:
        return {"relevant_docs": []}

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict technical document grader. Assess whether the retrieved document snippet "
            "contains relevant facts or code context to help answer the user question.\n"
            "Give a binary score: is_relevant: true or false, along with a brief rationale.\n"
            "Respond in JSON matching the schema."
        )),
        ("human", "User Question: {question}\n\nDocument Snippet:\n{content}")
    ])
    
    llm = get_llm(temperature=0.0)
    structured_grader = llm.with_structured_output(DocumentGrade)
    grading_chain = grade_prompt | structured_grader

    relevant_docs = []
    for doc in docs:
        try:
            grade: DocumentGrade = grading_chain.invoke({
                "question": question,
                "content": doc["content"][:1000]
            })
            if grade.is_relevant:
                relevant_docs.append(doc)
        except Exception:
            # On parser fallback, keep document if score is high
            if doc.get("score", 0.0) > 0.35:
                relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}

def query_rewrite_node(state: GraphState) -> Dict[str, Any]:
    """
    Fallback loop node: If all docs were irrelevant, reformulate query and increment retry_count.
    """
    current_retry = state.get("retry_count", 0) + 1
    question = state["question"]
    prev_query = state.get("transformed_query", question)

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a search query optimizer. The previous query failed to retrieve relevant technical documents.\n"
            "Look at the user's initial question and the previous query, and formulate an alternative, "
            "clearer, keyword-rich query specifically tailored for technical documentation."
        )),
        ("human", "Original Question: {question}\nPrevious Query: {prev_query}\n\nProvide ONLY the reformulated search query.")
    ])

    try:
        llm = get_llm(temperature=0.2)
        chain = rewrite_prompt | llm | StrOutputParser()
        new_query = chain.invoke({"question": question, "prev_query": prev_query}).strip()
    except Exception:
        new_query = f"{question} documentation implementation details"

    return {
        "transformed_query": new_query,
        "retry_count": current_retry
    }

def web_fallback_node(state: GraphState) -> Dict[str, Any]:
    """
    Bonus Node: If vector retrieval fails after max retries, search web (DuckDuckGo).
    """
    question = state["question"]
    fallback_docs = []
    
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=3))
            for r in results:
                fallback_docs.append({
                    "content": f"{r.get('title', '')}: {r.get('body', '')}",
                    "source": r.get("href", "DuckDuckGo Web Search"),
                    "section": r.get("title", "Web Result"),
                    "score": 0.5
                })
    except Exception as e:
        pass

    return {
        "relevant_docs": fallback_docs,
        "web_fallback_used": True
    }

def generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 4: Synthesizes final answer strictly grounded in relevant documents with citations.
    """
    question = state["question"]
    docs = state.get("relevant_docs", [])
    
    if not docs:
        return {
            "generation": "I'm sorry, but the available technical documentation does not contain sufficient information to answer your question accurately.",
            "citations": []
        }

    # Format context with explicit citation anchors
    context_blocks = []
    citations = []
    
    for i, doc in enumerate(docs):
        source = doc.get("source", "doc")
        section = doc.get("section", "")
        citation_label = f"[{source} | {section}]" if section else f"[{source}]"
        citations.append(citation_label)
        context_blocks.append(f"--- Document Source {i+1}: {citation_label} ---\n{doc['content']}\n")

    context_str = "\n".join(context_blocks)

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a technical documentation assistant. Answer the user question using ONLY the provided context.\n"
            "Rules:\n"
            "1. Ground your answer strictly in the provided documentation.\n"
            "2. Include inline citations to the sources in brackets (e.g. `[fastapi_dependencies.md | Sub-dependencies]`).\n"
            "3. If code is applicable, provide clean, idiomatic code examples based on the docs.\n"
            "4. If the context does not contain the answer, state that context is insufficient."
        )),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])

    llm = get_llm(temperature=0.1)
    chain = gen_prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context_str})

    return {
        "generation": answer,
        "citations": list(set(citations))
    }
