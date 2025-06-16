from typing import List, Dict, Any, Optional, TypedDict

"""
Ten moduł zawiera definicje typów (schematów) dla kluczowych
struktur danych używanych w całej aplikacji.

Użycie `total=False` pozwala na elastyczność, oznaczając,
że nie wszystkie klucze muszą być obecne w każdym słowniku.
"""

class ContextData(TypedDict, total=False):
    context_id: str
    origin_source_id: str
    context_content: str
    generated_by: str
    generation_date: str

class QuestionData(TypedDict, total=False):
    question_id: str
    category: str
    question_type: str
    origin_source_id: str
    context_ids: List[str]
    question_text: str
    choices: Optional[List[Dict[str, str]]]
    answer: Optional[Dict[str, Any]]
    generated_by: str
    generation_date: str

class ModelAnswerData(TypedDict, total=False):
    model_answer_id: str
    question_id: str
    model_answer_raw_text: str
    model_answer_clean_text: str
    generated_by: str
    generation_date: str
    model_configuration: str
