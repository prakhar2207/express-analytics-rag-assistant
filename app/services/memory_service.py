from typing import Dict, List, Optional
import time

class SessionMemoryService:
    """
    In-memory session conversation manager for multi-turn technical Q&A.
    Tracks dialogue history to resolve follow-up questions and coreferences.
    """
    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        if not session_id or session_id not in self.sessions:
            return []
        return self.sessions[session_id]

    def add_interaction(self, session_id: str, question: str, answer: str) -> None:
        if not session_id:
            return
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "question": question,
            "answer": answer,
            "timestamp": str(time.time())
        })
        
        # Keep last N turns to avoid context overflow
        if len(self.sessions[session_id]) > self.max_history_turns:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_turns:]

    def format_history_for_prompt(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return "No previous conversation."
        
        lines = []
        for turn in history:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer'][:200]}...")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

memory_service = SessionMemoryService()
