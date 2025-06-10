from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Context:
    context_id: str
    context_content: str
    origin_source_id: Optional[str] = None
    generated_by: Optional[str] = None
    generation_date: Optional[str] = None


@dataclass
class Question:
    question_id: str
    question_text: str
    question_type: str
    category: str
    origin_source_id: Optional[str] = None
    generated_by: Optional[str] = None
    generation_date: Optional[str] = None

    contexts: List[Context] = field(default_factory=list)
    choices: Optional[List[Dict[str, str]]] = None
    answer: Optional[Dict[str, Any]] = None

    @property
    def context_ids(self) -> List[str]:
        return [ctx.context_id for ctx in self.contexts]