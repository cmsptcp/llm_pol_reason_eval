import pytest
import tempfile
import json
from pathlib import Path
from llm_pol_reason_eval.qa_engine.llm_qa_engine import LLMQAEngine
from llm_pol_reason_eval.qa_engine.inference_client import InferenceClient


@pytest.fixture
def mock_inference_client(mocker):
    mock_client = mocker.MagicMock(spec=InferenceClient)

    def mock_get_response(prompt: str, **kwargs):
        num_questions = prompt.count("### PYTANIE")
        answers = [f"Mocked Answer {i + 1}" for i in range(num_questions)]
        separator = "[[---KONIEC ODPOWIEDZI---]]"
        return separator.join(answers) + separator

    mock_client.get_response.side_effect = mock_get_response
    return mock_client


@pytest.fixture
def temp_dataset_file():
    test_data = {
        "contexts": {"C1": {"context_id": "C1", "context_content": "Content C1"}},
        "questions": {
            "Q1": {"question_id": "Q1", "category": "cat1", "question_type": "type1", "question_text": "Text Q1",
                   "context_ids": ["C1"]},
            "Q2": {"question_id": "Q2", "category": "cat1", "question_type": "type1", "question_text": "Text Q2",
                   "context_ids": []}
        }
    }
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json", encoding='utf-8') as tmp:
        json.dump(test_data, tmp)
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink()


def test_llm_qa_engine_generate_answers(mock_inference_client, temp_dataset_file):
    model_name = "bielik-1.5b-v3-instruct"
    model_path = "speakleash/Bielik-1.5B-v3.0-Instruct"

    engine = LLMQAEngine(model_name, model_path, mock_inference_client)
    engine.generate_answers(dataset_filepath=temp_dataset_file, batch_size=10)

    assert len(engine.results) == 2
    mock_inference_client.get_response.assert_called_once()

    result_q1 = next(r for r in engine.results if r["question_id"] == "Q1")
    result_q2 = next(r for r in engine.results if r["question_id"] == "Q2")

    assert result_q1["answer_text"] == "Mocked Answer 1"
    assert result_q2["answer_text"] == "Mocked Answer 2"
    assert result_q1["generated_by"].startswith(model_name)
