from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Context:
    context_id: str
    context_content: str
    origin_source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    question_id: str
    question_text: str
    question_type: str
    category: str
    contexts: List[Context] = field(default_factory=list)
    choices: Optional[List[Dict[str, str]]] = None
    answer: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    @property
    def context_ids(self) -> List[str]:
        return [ctx.context_id for ctx in self.contexts]