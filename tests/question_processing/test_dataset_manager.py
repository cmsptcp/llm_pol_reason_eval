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
                 "statement_evaluations"],
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
                assert req_field in context, f"Brak wymaganego pola '{req_field}' w kontekście {cid}"

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


@pytest.mark.parametrize('json_paths', [get_generated_json_files()])
def test_dataset_manager(json_paths):
    if not json_paths:
        pytest.skip("Nie znaleziono plików JSON do przetestowania.")

    ds = DatasetManager()

    # 1) Załadować wszystkie pliki json do datasetu
    for json_path in json_paths:
        ds.add_data_from_json_file(json_path, duplicate_strategy="replace")  # Użyj replace, aby mieć pewność stanu

    # 2) Sprawdzić, czy działają statystyki datasetu
    initial_question_count = len(ds.questions)
    initial_context_count = len(ds.contexts)
    print(f"Załadowano {initial_question_count} pytań i {initial_context_count} kontekstów.")

    # 3) Sprawdzić, czy dataset ma co najmniej 1 pytanie
    assert initial_question_count > 0, "Dataset powinien zawierać co najmniej jedno pytanie."
    # 4) Sprawdzić, czy dataset ma co najmniej 1 kontekst
    assert initial_context_count > 0, "Dataset powinien zawierać co najmniej jeden kontekst."

    # 5) Spróbować dodać jeszcze raz każdy z plików w różnych strategiach
    # Użyjemy pierwszego pliku do testowania strategii na głównym ds
    if json_paths:
        first_json_path = json_paths[0]

        # Strategia "skip"
        count_q_before_skip = len(ds.questions)
        count_c_before_skip = len(ds.contexts)
        ds.add_data_from_json_file(first_json_path, duplicate_strategy="skip")
        assert len(
            ds.questions) == count_q_before_skip, "Strategia 'skip' nie powinna zmieniać liczby pytań, jeśli dane już istnieją."
        assert len(
            ds.contexts) == count_c_before_skip, "Strategia 'skip' nie powinna zmieniać liczby kontekstów, jeśli dane już istnieją."

        # Strategia "replace"
        ds.add_data_from_json_file(first_json_path, duplicate_strategy="replace")
        temp_ds_replace = DatasetManager()
        temp_ds_replace.add_data_from_json_file(first_json_path)
        assert len(ds.questions) >= len(temp_ds_replace.questions) or not temp_ds_replace.questions
        assert len(ds.contexts) >= len(temp_ds_replace.contexts) or not temp_ds_replace.contexts

        # Strategia "merge"
        ds.add_data_from_json_file(first_json_path, duplicate_strategy="merge")
        assert len(ds.questions) >= len(temp_ds_replace.questions) or not temp_ds_replace.questions
        assert len(ds.contexts) >= len(temp_ds_replace.contexts) or not temp_ds_replace.contexts

    # 6) Pobrać przykładowe pytanie
    sample_question_id = None
    sample_question_data = None
    if ds.questions:
        sample_question_id = list(ds.questions.keys())[0]
        sample_question_data = ds.questions[sample_question_id]

        # 7) Usunąć to pytanie
        ds.remove_questions_by_query(lambda q: q['question_id'] == sample_question_id)
        assert sample_question_id not in ds.questions, "Przykładowe pytanie nie zostało usunięte."

    # 8) Usunąć pytanie, którego nie ma w datasecie
    count_before_remove_non_existent = len(ds.questions)
    ds.remove_questions_by_query(lambda q: q['question_id'] == "ID_KTORE_NA_PEWNO_NIE_ISTNIEJE_12345")
    assert len(
        ds.questions) == count_before_remove_non_existent, "Usunięcie nieistniejącego pytania nie powinno zmieniać liczby pytań."

    # 9) Dodać z powrotem pytanie które wcześniej usunięto
    if sample_question_id and sample_question_data:
        question_to_add_back_str = json.dumps({"questions": {sample_question_id: sample_question_data}})
        ds.add_data_from_json_string(question_to_add_back_str, duplicate_strategy="replace")
        assert sample_question_id in ds.questions, "Przykładowe pytanie nie zostało dodane z powrotem."

    # 10) Spróbować usunąć kontekst, który jest powiązany z jakimś pytaniem
    linked_context_id_to_remove = None
    if ds.questions and ds.contexts:
        for q_id, q_data in ds.questions.items():
            if q_data.get("context_ids"):
                for ctx_id in q_data["context_ids"]:
                    if ctx_id in ds.contexts:
                        linked_context_id_to_remove = ctx_id
                        break
            if linked_context_id_to_remove:
                break

    if linked_context_id_to_remove:
        print(f"Próba usunięcia powiązanego kontekstu: {linked_context_id_to_remove}")
        # Oczekujemy błędu, ponieważ kontekst jest powiązany z pytaniem
        with pytest.raises(ValueError,
                           match=f"Kontekst {linked_context_id_to_remove} jest powiązany z co najmniej jednym pytaniem i nie może zostać usunięty."):
            ds.remove_contexts_by_query(lambda c: c['context_id'] == linked_context_id_to_remove)
        # Sprawdzenie, czy kontekst nadal istnieje po nieudanej próbie usunięcia
        assert linked_context_id_to_remove in ds.contexts, "Powiązany kontekst nie powinien zostać usunięty, jeśli zgłoszono błąd."
    else:
        print("Nie znaleziono kontekstu powiązanego z pytaniem do przetestowania usunięcia.")

    # 11) Pobrać pytania w batchach po 3
    if ds.questions:
        batches_of_3_str_list = ds.get_questions_in_batches_as_jsonl_string(batch_size=3)
        assert isinstance(batches_of_3_str_list, list), "Wynik batchowania powinien być listą."
        if batches_of_3_str_list:  # Jeśli są jakiekolwiek batche
            assert isinstance(batches_of_3_str_list[0], str), "Elementy listy batchy powinny być stringami JSONL."
            first_batch_data = json.loads(batches_of_3_str_list[0])
            assert "questions" in first_batch_data, "Każdy batch powinien zawierać klucz 'questions'."
            assert len(first_batch_data["questions"]) <= 3, "Rozmiar batcha nie powinien przekraczać zadanego."

    # 12) Pobrać pytania 1 typu z kontekstami
    # Znajdź pierwszy dostępny typ pytania
    example_question_type = None
    if ds.questions:
        for q_data in ds.questions.values():
            if "question_type" in q_data:
                example_question_type = q_data["question_type"]
                break

    if example_question_type:
        typed_questions_str = ds.get_questions_as_json_string(
            query=lambda q: q.get("question_type") == example_question_type,
            with_contexts=True
        )
        typed_questions_data = json.loads(typed_questions_str)
        assert "questions" in typed_questions_data
        if typed_questions_data["questions"]:  # Jeśli znaleziono pytania tego typu
            assert "contexts" in typed_questions_data, "Konteksty powinny być dołączone, jeśli with_contexts=True i są pytania."
            for q_data in typed_questions_data["questions"].values():
                assert q_data.get("question_type") == example_question_type

    # 13) Pobrać pytania 1 typu w batchach po 5 z kontekstami
    if example_question_type and ds.questions:
        typed_batches_str_list = ds.get_questions_in_batches_as_jsonl_string(
            batch_size=5,
            query=lambda q: q.get("question_type") == example_question_type,
            with_contexts=True,
            sort_key="question_id"  # Dodajemy sortowanie dla spójności
        )
        assert isinstance(typed_batches_str_list, list)
        if typed_batches_str_list:
            first_typed_batch_data = json.loads(typed_batches_str_list[0])
            assert "questions" in first_typed_batch_data
            if first_typed_batch_data["questions"]:
                assert "contexts" in first_typed_batch_data
                assert len(first_typed_batch_data["questions"]) <= 5
                for q_data_batch in first_typed_batch_data["questions"].values():
                    assert q_data_batch.get("question_type") == example_question_type

    # 14) Zapisać plik jsonl
    temp_dir = tempfile.mkdtemp()
    try:
        output_jsonl_path = os.path.join(temp_dir, "test_batch_output.jsonl")
        ds.save_questions_in_batches_as_jsonl_file(
            filepath=output_jsonl_path,
            batch_size=2,  # Mały batch dla testu
            query=lambda q: True,  # Wszystkie pytania
            with_contexts=True,
            sort_key="question_id"
        )
        assert os.path.exists(output_jsonl_path), "Plik JSONL nie został utworzony."

        # 15) Otworzyć plik jsonl i spróbować dodać do nowego datasetu wybrany batch
        if os.path.exists(output_jsonl_path) and os.path.getsize(output_jsonl_path) > 0:
            with open(output_jsonl_path, "r", encoding="utf-8") as f:
                first_line_batch_str = f.readline().strip()

            if first_line_batch_str:
                ds_new = DatasetManager()
                ds_new.add_data_from_json_string(first_line_batch_str)

                first_batch_json_data = json.loads(first_line_batch_str)
                expected_q_count_in_batch = len(first_batch_json_data.get("questions", {}))
                expected_c_count_in_batch = len(first_batch_json_data.get("contexts", {}))

                assert len(
                    ds_new.questions) == expected_q_count_in_batch, "Liczba pytań w nowym DS nie zgadza się z batchem."
                assert len(
                    ds_new.contexts) == expected_c_count_in_batch, "Liczba kontekstów w nowym DS nie zgadza się z batchem."
            else:
                print("Plik JSONL był pusty, pomijanie testu ładowania batcha.")
        else:
            print("Plik JSONL nie istnieje lub jest pusty, pomijanie testu ładowania batcha.")
            if not ds.questions:  # Jeśli nie było pytań, plik JSONL może być pusty
                assert not os.path.exists(output_jsonl_path) or os.path.getsize(output_jsonl_path) == 0

    finally:
        shutil.rmtree(temp_dir)

    # Końcowe sprawdzenie zapisu i odczytu całego datasetu
    final_temp_dir = tempfile.mkdtemp()
    try:
        all_data_path = os.path.join(final_temp_dir, "all_data.json")
        ds.save_all_data_to_json_file(all_data_path)
        assert os.path.exists(all_data_path)

        ds_loaded_all = DatasetManager()
        ds_loaded_all.add_data_from_json_file(all_data_path)
        assert len(ds_loaded_all.questions) == len(ds.questions)
        assert len(ds_loaded_all.contexts) == len(ds.contexts)
    finally:
        shutil.rmtree(final_temp_dir)