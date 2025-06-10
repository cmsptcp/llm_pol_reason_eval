import json
import os
import glob
import tempfile
import shutil
import pytest
from llm_pol_reason_eval.question_processing.dataset_manager import DatasetManager

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
    "optional": ["scoring_rules", "exam_requirements", "external_context_required", "correct_answer", "example_answers",
                 "statement_evaluations"]
}
DATASET_DIR = 'data/dataset_raw'


def assert_allowed_keys(data, data_fields, entity_name):
    allowed_keys = set(data_fields["required"]) | set(data_fields.get("optional", []))
    data_keys = set(data.keys())
    assert data_keys.issubset(allowed_keys), f"{entity_name} zawiera niedozwolone pola: {data_keys - allowed_keys}"
    for req_field in data_fields["required"]:
        assert req_field in data, f"{entity_name} nie zawiera wymaganego pole '{req_field}'"


def get_generated_json_files():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return glob.glob(os.path.join(base_dir, DATASET_DIR, '**', '*.json'), recursive=True)


@pytest.mark.parametrize('json_path', get_generated_json_files())
def test_generated_question_json_structure(json_path):
    # Ten test pozostaje bez zmian
    with open(json_path, encoding='utf-8') as f:
        json_data = json.load(f)
    if 'contexts' in json_data:
        for cid, context in json_data["contexts"].items():
            assert isinstance(context, dict)
            assert cid == context.get("context_id")
            assert_allowed_keys(context, CONTEXT_FIELDS, f"Kontekst {cid}")
    assert "questions" in json_data
    for qid, question in json_data["questions"].items():
        assert isinstance(question, dict)
        assert qid == question.get("question_id")
        assert_allowed_keys(question, QUESTION_FIELDS, f"Pytanie {qid}")
        assert "answer" in question
        assert_allowed_keys(question["answer"], ANSWER_FIELDS, f"Odpowiedź w pytaniu {qid}")


@pytest.mark.parametrize('json_paths', [get_generated_json_files()])
def test_dataset_manager_integration(json_paths):
    if not json_paths:
        pytest.skip("Nie znaleziono plików JSON do przetestowania.")

    ds = DatasetManager()

    # 1) Załadować wszystkie pliki json do datasetu
    for json_path in json_paths:
        ds.add_data_from_json_file(json_path, duplicate_strategy="replace")

    # 2) Sprawdzić, czy działają statystyki datasetu (wstępne wywołanie)
    initial_question_count = len(ds.questions)
    initial_context_count = len(ds.contexts)
    print(f"Załadowano {initial_question_count} pytań i {initial_context_count} kontekstów.")

    # 3) Sprawdzić, czy dataset ma co najmniej 1 pytanie
    assert initial_question_count > 0

    # 4) Sprawdzić, czy dataset ma co najmniej 1 kontekst
    assert initial_context_count > 0

    # 5) Spróbować dodać jeszcze raz plik w strategii "skip"
    first_json_path = json_paths[0]
    count_q_before_skip = len(ds.questions)
    ds.add_data_from_json_file(first_json_path, duplicate_strategy="skip")
    assert len(ds.questions) == count_q_before_skip

    # 6) Pobrać przykładowe pytanie
    sample_question_id = list(ds.questions.keys())[0]
    sample_question_data = ds.questions[sample_question_id]

    # 7) Usunąć to pytanie
    ds.remove_questions_by_query(lambda q: q['question_id'] == sample_question_id)
    assert sample_question_id not in ds.questions

    # 8) Usunąć pytanie, którego nie ma w datasecie
    count_before_remove_non_existent = len(ds.questions)
    ds.remove_questions_by_query(lambda q: q['question_id'] == "ID_KTOREGO_NA_PEWNO_NIE_MA")
    assert len(ds.questions) == count_before_remove_non_existent

    # 9) Dodać z powrotem pytanie, które wcześniej usunięto
    question_to_add_back_str = json.dumps({"questions": {sample_question_id: sample_question_data}})
    ds.add_data_from_json_string(question_to_add_back_str)
    assert sample_question_id in ds.questions

    # 10) Spróbować usunąć kontekst, który jest powiązany z jakimś pytaniem
    linked_context_id = next(
        (cid for q in ds.questions.values() for cid in q.get("context_ids", []) if cid in ds.contexts), None)
    if linked_context_id:
        with pytest.raises(ValueError):
            ds.remove_contexts_by_query(lambda c: c['context_id'] == linked_context_id, force=False)
        assert linked_context_id in ds.contexts
        ds.remove_contexts_by_query(lambda c: c['context_id'] == linked_context_id, force=True)
        assert linked_context_id not in ds.contexts

    # 11) Pobrać pytania w batchach (test ogólny)
    batches_str_list_generic = list(ds.get_grouped_question_batches(batch_size=3))
    assert isinstance(batches_str_list_generic, list)
    if batches_str_list_generic:
        assert isinstance(batches_str_list_generic[0], dict)

    # 12) Pobrać pytania 1 typu z kontekstami
    example_question_type = ds.get_question_type_list()[0] if ds.get_question_type_list() else None
    if example_question_type:
        typed_questions_str = ds.get_questions_as_json_string(
            query=lambda q: q.get("question_type") == example_question_type)
        assert isinstance(json.loads(typed_questions_str), dict)

    # 13) Pobrać pytania 1 typu w batchach po 5 z kontekstami
    if example_question_type:
        typed_batches_str_list = ds.get_grouped_question_batches_as_json_strings(
            batch_size=5, query=lambda q: q.get("question_type") == example_question_type)
        assert isinstance(typed_batches_str_list, list)

    # 14) Zapisać plik jsonl
    temp_dir = tempfile.mkdtemp()
    try:
        output_jsonl_path = os.path.join(temp_dir, "test_output.jsonl")
        ds.save_grouped_batches_as_jsonl_file(filepath=output_jsonl_path, batch_size=2)
        assert os.path.exists(output_jsonl_path)

        # 15) Otworzyć plik jsonl i spróbować dodać do nowego datasetu
        if os.path.getsize(output_jsonl_path) > 0:
            with open(output_jsonl_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            ds_new = DatasetManager()
            ds_new.add_data_from_json_string(first_line)
            assert len(ds_new.questions) > 0

    finally:
        shutil.rmtree(temp_dir)

    # 16) Sprawdzić poprawność działania metody get_stats()
    stats = ds.get_stats()
    assert isinstance(stats, dict)
    assert "question_category_stats" in stats
    assert "question_type_stats" in stats
    assert isinstance(stats["question_category_stats"], list)
    if stats["question_category_stats"]:
        assert isinstance(stats["question_category_stats"][0], tuple)
        assert len(stats["question_category_stats"][0]) == 2
        assert isinstance(stats["question_category_stats"][0][0], str)
        assert isinstance(stats["question_category_stats"][0][1], int)

    # 17) Sprawdzić poprawność działania metod get_question_category_list() i get_question_type_list()
    categories = ds.get_question_category_list()
    types = ds.get_question_type_list()
    assert isinstance(categories, list)
    assert isinstance(types, list)
    if categories:
        assert all(isinstance(cat, str) for cat in categories)
        assert categories == sorted(categories)  # Sprawdzenie, czy lista jest posortowana
    if types:
        assert all(isinstance(t, str) for t in types)
        assert types == sorted(types)

    # 18) Sprawdzić generator get_grouped_question_batches()
    generator = ds.get_grouped_question_batches(batch_size=1, query=lambda q: q["question_id"] == sample_question_id)
    first_and_only_batch = next(generator, None)
    assert first_and_only_batch is not None, "Generator powinien zwrócić co najmniej jeden batch dla istniejącego pytania."
    assert "questions" in first_and_only_batch
    assert "metadata" in first_and_only_batch
    assert sample_question_id in first_and_only_batch["questions"]
    assert first_and_only_batch["metadata"]["category"] == sample_question_data["category"]
    # Sprawdzenie, czy generator się wyczerpał
    assert next(generator, None) is None, "Generator powinien zwrócić dokładnie jeden batch."

    # 19) Zapis i odczyt całego datasetu
    final_temp_dir = tempfile.mkdtemp()
    try:
        all_data_path = os.path.join(final_temp_dir, "all_data.json")
        ds.save_all_data_to_json_file(all_data_path)
        ds_loaded_all = DatasetManager()
        ds_loaded_all.add_data_from_json_file(all_data_path)
        assert len(ds_loaded_all.questions) == len(ds.questions)
        assert len(ds_loaded_all.contexts) == len(ds.contexts)
    finally:
        shutil.rmtree(final_temp_dir)