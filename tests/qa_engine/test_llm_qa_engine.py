import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Ustaw katalog temp na poziomie projektu
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Patch AutoTokenizer przed importem LLMQAEngine
mock_tokenizer = MagicMock()
mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
sys.modules["transformers"] = MagicMock()
sys.modules["transformers"].AutoTokenizer = MagicMock(from_pretrained=lambda *a, **k: mock_tokenizer)

from llm_pol_reason_eval.qa_engine.llm_qa_engine import LLMQAEngine
from llm_pol_reason_eval.qa_engine.inference_client import InferenceClient
from llm_pol_reason_eval.question_processing.dataset_manager import DatasetManager

class MockInferenceClient(InferenceClient):
    def get_response(self, prompt_string, generation_params_override=None):
        return "<answer>Odpowiedź testowa</answer> [[---KONIEC ODPOWIEDZI---]]"

@pytest.fixture
def engine():
    dm = DatasetManager()
    dm.questions["q1"] = {
        "question_id": "q1",
        "question_text": "Ile to jest 2+2?",
        "question_type": "open_text",
        "context_ids": [],
        "category": "matematyka"
    }
    dm.questions["q2"] = {
        "question_id": "q2",
        "question_text": "Wymień stolicę Polski.",
        "question_type": "open_text",
        "context_ids": [],
        "category": "geografia"
    }
    engine = LLMQAEngine(
        model_name="mock-model",
        model_path="mock-path",
        inference_client=MockInferenceClient()
    )
    engine.dataset_manager = dm
    return engine

def test_generate_prompt_and_count_questions(engine):
    batch = next(engine.dataset_manager.get_grouped_question_batches(batch_size=2))
    prompt = engine.prompt_manager.get_question_prompt(
        {"name": "mock-model", "family": "default"},
        batch,
        {"main_template": "base_question_prompt.jinja2"}
    )
    prompt_text = "\n".join([m["content"] for m in prompt])
    zadanie_count = prompt_text.count("### ZADANIE")
    expected_count = len(batch["questions"])
    assert zadanie_count == expected_count
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text

def test_inference_and_answer_format(engine):
    batch = next(engine.dataset_manager.get_grouped_question_batches(batch_size=1))
    prompt = engine.prompt_manager.get_question_prompt(
        {"name": "mock-model", "family": "default"},
        batch,
        {"main_template": "base_question_prompt.jinja2"}
    )
    prompt_string = "\n".join([m["content"] for m in prompt])
    response = engine.inference_client.get_response(prompt_string)
    assert "<answer>" in response and "</answer>" in response
    assert "Odpowiedź testowa" in response

def test_full_pipeline(engine):
    batch = next(engine.dataset_manager.get_grouped_question_batches(batch_size=1))
    prompt = engine.prompt_manager.get_question_prompt(
        {"name": "mock-model", "family": "default"},
        batch,
        {"main_template": "base_question_prompt.jinja2"}
    )
    prompt_string = "\n".join([m["content"] for m in prompt])
    response = engine.inference_client.get_response(prompt_string)
    import re
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    assert match
    answer_text = match.group(1).strip()
    assert answer_text == "Odpowiedź testowa"

def test_dataset_manager_stats(engine):
    stats = engine.dataset_manager.get_stats()
    assert "question_category_stats" in stats
    assert ("matematyka", 1) in stats["question_category_stats"]
    assert ("geografia", 1) in stats["question_category_stats"]

def test_prompt_contains_no_ids(engine):
    batch = next(engine.dataset_manager.get_grouped_question_batches(batch_size=2))
    prompt = engine.prompt_manager.get_question_prompt(
        {"name": "mock-model", "family": "default"},
        batch,
        {"main_template": "base_question_prompt.jinja2"}
    )
    prompt_text = "\n".join([m["content"] for m in prompt])
    for q in batch["questions"].values():
        assert q.get("question_id", "") not in prompt_text
