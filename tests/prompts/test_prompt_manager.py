import pytest
import json
from pathlib import Path

from llm_pol_reason_eval.prompts.prompt_manager import PromptManager
from llm_pol_reason_eval.question_processing.dataset_manager import DatasetManager

# Ścieżki do plików testowych
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATASET_PATH = PROJECT_ROOT / "data/dataset/MPOP-P1-100-2412-gemini25pro-2025-06-14T20-55-00Z.json"
TEMPLATES_DIR = PROJECT_ROOT / "src/llm_pol_reason_eval/prompts/templates"

# Przykładowe konfiguracje modeli
QWEN_CFG = {
    "name": "qwen3-1-7b",
    "family": "qwen3"
}
BIELIK_CFG = {
    "name": "bielik-1-5b-v3-instruct",
    "family": "bielik"
}
NONEXISTENT_CFG = {
    "name": "nonexistent-model-123B-instruct",
    "family": "nonexistent"
}

@pytest.fixture(scope="module")
def dataset_manager():
    dm = DatasetManager()
    dm.add_data_from_json_file(str(DATASET_PATH))
    return dm

@pytest.fixture(scope="module")
def prompt_manager():
    return PromptManager(templates_dir=TEMPLATES_DIR)

def extract_prompt_text(messages):
    # Zwraca połączoną treść promptu (system + user)
    return "\n".join([m["content"] for m in messages])

def test_prompt_closed_MTF_qwen(dataset_manager, prompt_manager):
    # 1 pytanie closed_MTF, qwen3-1-7b
    batches = list(dataset_manager.get_grouped_question_batches(
        batch_size=1, query=lambda q: q["question_type"] == "closed_MTF"))
    batch = batches[0]
    prompt = prompt_manager.get_question_prompt(QWEN_CFG, batch, {"main_template": "base_question_prompt.jinja2"})
    prompt_text = extract_prompt_text(prompt)
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text
        for cid in q.get("context_ids", []):
            if cid in batch.get("contexts", {}):
                assert batch["contexts"][cid]["context_content"] in prompt_text
        assert q.get("question_id", "") not in prompt_text
        for cid in q.get("context_ids", []):
            assert cid not in prompt_text

def test_prompt_open_text_5_qwen(dataset_manager, prompt_manager):
    # 5 pytań open_text, qwen3-1-7b
    batches = list(dataset_manager.get_grouped_question_batches(
        batch_size=5, query=lambda q: q["question_type"] == "open_text"))
    batch = batches[0]
    prompt = prompt_manager.get_question_prompt(QWEN_CFG, batch, {"main_template": "base_question_prompt.jinja2"})
    prompt_text = extract_prompt_text(prompt)
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text
        for cid in q.get("context_ids", []):
            if cid in batch.get("contexts", {}):
                assert batch["contexts"][cid]["context_content"] in prompt_text
        assert q.get("question_id", "") not in prompt_text
        for cid in q.get("context_ids", []):
            assert cid not in prompt_text

def test_prompt_open_text_1_qwen(dataset_manager, prompt_manager):
    # 1 pytanie open_text, qwen3-1-7b
    batches = list(dataset_manager.get_grouped_question_batches(
        batch_size=1, query=lambda q: q["question_type"] == "open_text"))
    batch = batches[0]
    prompt = prompt_manager.get_question_prompt(QWEN_CFG, batch, {"main_template": "base_question_prompt.jinja2"})
    prompt_text = extract_prompt_text(prompt)
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text
        for cid in q.get("context_ids", []):
            if cid in batch.get("contexts", {}):
                assert batch["contexts"][cid]["context_content"] in prompt_text
        assert q.get("question_id", "") not in prompt_text
        for cid in q.get("context_ids", []):
            assert cid not in prompt_text

def test_prompt_open_synthesis_1_qwen(dataset_manager, prompt_manager):
    # 1 pytanie open_synthesis, qwen3-1-7b
    batches = list(dataset_manager.get_grouped_question_batches(
        batch_size=1, query=lambda q: q["question_type"] == "open_synthesis"))
    batch = batches[0]
    prompt = prompt_manager.get_question_prompt(QWEN_CFG, batch, {"main_template": "base_question_prompt.jinja2"})
    prompt_text = extract_prompt_text(prompt)
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text
        for cid in q.get("context_ids", []):
            if cid in batch.get("contexts", {}):
                assert batch["contexts"][cid]["context_content"] in prompt_text
        assert q.get("question_id", "") not in prompt_text
        for cid in q.get("context_ids", []):
            assert cid not in prompt_text

def test_prompt_closed_MTF_2_bielik(dataset_manager, prompt_manager):
    # 2 pytania closed_MTF, bielik-1-5b-v3-instruct
    batches = list(dataset_manager.get_grouped_question_batches(
        batch_size=2, query=lambda q: q["question_type"] == "closed_MTF"))
    batch = batches[0]
    prompt = prompt_manager.get_question_prompt(BIELIK_CFG, batch, {"main_template": "base_question_prompt.jinja2"})
    prompt_text = extract_prompt_text(prompt)
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text
        for cid in q.get("context_ids", []):
            if cid in batch.get("contexts", {}):
                assert batch["contexts"][cid]["context_content"] in prompt_text
        assert q.get("question_id", "") not in prompt_text
        for cid in q.get("context_ids", []):
            assert cid not in prompt_text

def test_prompt_open_text_1_bielik(dataset_manager, prompt_manager):
    # 1 pytanie open_text, bielik-1-5b-v3-instruct
    batches = list(dataset_manager.get_grouped_question_batches(
        batch_size=1, query=lambda q: q["question_type"] == "open_text"))
    batch = batches[0]
    prompt = prompt_manager.get_question_prompt(BIELIK_CFG, batch, {"main_template": "base_question_prompt.jinja2"})
    prompt_text = extract_prompt_text(prompt)
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text
        for cid in q.get("context_ids", []):
            if cid in batch.get("contexts", {}):
                assert batch["contexts"][cid]["context_content"] in prompt_text
        assert q.get("question_id", "") not in prompt_text
        for cid in q.get("context_ids", []):
            assert cid not in prompt_text

def test_prompt_open_text_1_nonexistent(dataset_manager, prompt_manager):
    # 1 pytanie open_text, model nieistniejący
    batches = list(dataset_manager.get_grouped_question_batches(
        batch_size=1, query=lambda q: q["question_type"] == "open_text"))
    batch = batches[0]
    prompt = prompt_manager.get_question_prompt(
        NONEXISTENT_CFG, batch, {"main_template": "base_question_prompt.jinja2"}
    )
    prompt_text = extract_prompt_text(prompt)
    # Sprawdź, że prompt zawiera treść pytania (czyli szablon default został użyty)
    for q in batch["questions"].values():
        assert q["question_text"] in prompt_text
