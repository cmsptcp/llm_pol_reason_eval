import json
import os
import glob
import tempfile
import shutil
import pytest
import time
from llm_pol_reason_eval.question_processing.dataset_manager import DatasetManager, DuplicatesStrategy


CONTEXT_FIELDS = {
    "required": ["context_id", "origin_source_id", "context_content"],
    "optional": ["generated_by", "generation_date"]
}

QUESTION_FIELDS = {
    "required": ["question_id", "category", "question_type", "origin_source_id", "context_ids", "question_text",
                 "answer"],
    "optional": ["generated_by", "generation_date", "choices"]
}

ANSWER_FIELDS = {
    "required": ["max_points"],
    "optional": ["scoring_rules", "exam_requirements", "external_context_required", "correct_answer", "example_answers", "statement_evaluations"],
    "required_any": set()
}

DATASET_DIR = 'data/dataset_raw'

def get_generated_json_files():
    return glob.glob(os.path.join(DATASET_DIR, '**', '*.json'), recursive=True)

@pytest.mark.parametrize('json_path', get_generated_json_files())
def test_generated_question_json_structure(json_path):
    with open(json_path, encoding='utf-8') as f:
        json_data = json.load(f)

    if 'contexts' in json_data:
        for cid, context in json_data["contexts"].items():
            assert isinstance(context, dict), f"Kontekst {cid} nie jest słownikiem."
            assert "context_id" in context, f"Kontekst {cid} nie zawiera pola 'context_id'."
            assert cid == context["context_id"], \
                f"Kontekst {cid} ma 'context_id' różne od klucza w słowniku: {context['context_id']}."

            allowed_context_keys = set(CONTEXT_FIELDS["required"]) | set(CONTEXT_FIELDS["optional"])
            context_keys = set(context.keys())

            assert context_keys.issubset(allowed_context_keys), \
                f"Kontekst {cid} zawiera niedozwolone pola: {context_keys - allowed_context_keys}"

            for req_field in CONTEXT_FIELDS["required"]:
                assert req_field in context, f"Brak wymaganego pola '{req_field}' w kontekście {i}"

    assert "questions" in json_data, "Brak klucza 'questions' w danych JSON."
    for qid, question in json_data["questions"].items():
        assert isinstance(question, dict), f"Pytanie {qid} nie jest słownikiem."
        assert "question_id" in question, f"Pytanie {qid} nie zawiera pola 'question_id'."
        assert qid == question["question_id"], \
            f"Pytanie {qid} ma 'question_id' różne od klucza w słowniku: {question['question_id']}."

        question_keys = set(question.keys())
        allowed_question_keys = set(QUESTION_FIELDS["required"]) | set(QUESTION_FIELDS["optional"])

        assert question_keys.issubset(allowed_question_keys), \
            f"Pytanie {qid} zawiera niedozwolone pola: {question_keys - allowed_question_keys}"

        for req_field in QUESTION_FIELDS["required"]:
            assert req_field in question, f"Brak wymaganego pola '{req_field}' w pytaniu {qid}"

        assert "answer" in question, f"Brak pola 'answer' w pytaniu {qid}"
        answer = question["answer"]
        answer_keys = set(answer.keys())
        allowed_answer_keys = set(ANSWER_FIELDS["required"]) | set(ANSWER_FIELDS["optional"])

        assert answer_keys.issubset(allowed_answer_keys), \
            f"Odpowiedź w pytaniu {qid} zawiera niedozwolone pola: {answer_keys - allowed_answer_keys}"

        for req_field in ANSWER_FIELDS["required"]:
            assert req_field in answer, f"Brak wymaganego pola '{req_field}' w odpowiedzi pytania {qid}"


@pytest.mark.parametrize('json_path', get_generated_json_files())
def test_dataset_processor_full(json_path):
    ds = DatasetManager()

    # Dodawanie danych
    ds.add_data_from_json_file(json_path)
    assert not ds.contexts_df.empty or not ds.questions_df.empty

    # Dodawanie z inną strategią duplikatów
    ds.add_data_from_json_file(json_path, dup_strategy=DuplicatesStrategy.IGNORE_DUPLICATES)
    assert ds.questions_df['question_id'].is_unique

    # Pobierz pytania typu closed_MTF jako JSON (z kontekstami)
    json_string_closed_mtf = ds.get_questions_as_json_string(
        query_string="question_type == 'closed_MTF'",
        with_contexts=True
    )
    closed_mtf_questions = json.loads(json_string_closed_mtf)["questions"]
    assert closed_mtf_questions, "Brak pytań closed_MTF"
    closed_mtf_ids = [q["question_id"] for q in closed_mtf_questions]
    first_closed_mtf_id = closed_mtf_ids[0]

    # Usuń jedno z tych pytań
    ds.remove_questions_by_query(f"question_id == '{first_closed_mtf_id}'")
    assert first_closed_mtf_id not in ds.questions_df['question_id'].values

    # Dodaj ponownie pytania closed_MTF z json_string_closed_mtf
    ds.add_data_from_json_string(json_string_closed_mtf)
    assert first_closed_mtf_id in ds.questions_df['question_id'].values

    # Usuwanie pytań po query
    initial_q = len(ds.questions_df)
    ds.remove_questions_by_query("question_type == 'closed_MTF'")
    assert len(ds.questions_df) <= initial_q

    # Usuwanie kontekstów po query (bez force)
    initial_c = len(ds.contexts_df)
    ds.remove_contexts_by_query("origin_source_id == 'nonexistent'")
    assert len(ds.contexts_df) <= initial_c

    # Usuwanie kontekstów po query (force)
    ds.remove_contexts_by_query("origin_source_id == 'nonexistent'", by_force=True)

    # Serializacja wszystkich danych do stringa
    all_json = ds.get_all_data_as_json_string()
    assert 'questions' in all_json

    # Serializacja pytań z query i kontekstami
    q_json = ds.get_questions_as_json_string(query_string="question_type == 'open'", with_contexts=True)
    parsed = json.loads(q_json)
    assert 'questions' in parsed
    assert 'contexts' in parsed

    # Serializacja kontekstów z query
    c_json = ds.get_contexts_as_json_string(query_string="origin_source_id == 'nonexistent'")
    assert 'contexts' in json.loads(c_json)

    # Zapis wszystkich danych do pliku
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmpfile:
        out_path = tmpfile.name
    ds.save_all_data_to_json_file(out_path)
    assert os.path.exists(out_path)
    with open(out_path, encoding='utf-8') as f:
        data = json.load(f)
    assert 'questions' in data
    os.remove(out_path)

    # Zapis batchy pytań do pliku jsonl
    temp_dir = tempfile.mkdtemp()
    output_jsonl = os.path.join(temp_dir, "batch.jsonl")
    ds.save_questions_in_batches_as_jsonl_file(
        output_jsonl,
        batch_size=2,
        query_string="question_id.str.contains('Zadanie_1', na=False) or question_id.str.contains('Zadanie_2', na=False)",
        sort_questions_by_key='question_id'
    )
    assert os.path.exists(output_jsonl)
    with open(output_jsonl, encoding='utf-8') as f:
        lines = f.readlines()
    assert lines

    # Pobranie batchy jako string jsonl
    jsonl_str = ds.get_questions_in_batches_as_jsonl_string(
        batch_size=2,
        query_string="question_id.str.contains('Zadanie_1', na=False)",
        with_contexts=True,
        sort_questions_by_key='question_id'
    )
    assert jsonl_str.strip() != ""

    shutil.rmtree(temp_dir)