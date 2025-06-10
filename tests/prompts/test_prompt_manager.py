import pytest
from pathlib import Path
from llm_pol_reason_eval.prompts.prompt_manager import PromptManager

TEMPLATES_DIR = r"C:\Users\piotr\PycharmProjects\LLMPolReasonEval\src\llm_pol_reason_eval\prompts\templates"
#Path(__file__).resolve().parents[2] / "src/llm_pol_reason_eval/prompts/templates"

@pytest.fixture
def prompt_manager() -> PromptManager:
    return PromptManager(templates_dir=TEMPLATES_DIR)

def test_get_question_prompt(prompt_manager):
    model_name = "bielik-1-5b-v3-instruct"
    batch_data = {
        "questions": {"Q1": {"question_text": "Tekst pytania", "context_ids": ["C1"]}},
        "contexts": {"C1": {"context_content": "Treść kontekstu"}},
    }
    messages = prompt_manager.get_question_prompt(model_name, batch_data)
    assert len(messages) == 2
    assert messages[1]['role'] == 'user'
    assert "### PYTANIE (ID: Q1)" in messages[1]['content']
    assert "Treść kontekstu" in messages[1]['content']