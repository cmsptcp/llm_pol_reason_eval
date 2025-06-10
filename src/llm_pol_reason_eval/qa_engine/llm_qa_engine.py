import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from transformers import AutoTokenizer

from llm_pol_reason_eval.question_processing.dataset_manager import DatasetManager
from llm_pol_reason_eval.prompts.prompt_manager import PromptManager
from llm_pol_reason_eval.qa_engine.question import Question, Context


class LLMQAEngine:

    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.dataset_manager = DatasetManager()

        templates_path = Path(__file__).parent.parent / "prompts" / "templates"
        self.prompt_manager = PromptManager(templates_dir=templates_path)

        self.results: List[Dict[str, Any]] = []

    def _create_question_object(self, q_id: str, q_data: Dict[str, Any]) -> Question:
        contexts_data = [
            self.dataset_manager.contexts.get(cid) for cid in q_data.get("context_ids", [])
        ]

        contexts = [
            Context(
                context_id=ctx.get("context_id"),
                context_content=ctx.get("context_content"),
                origin_source_id=ctx.get("origin_source_id")
            )
            for ctx in contexts_data if ctx is not None
        ]

        return Question(
            question_id=q_id,
            question_text=q_data.get("question_text"),
            question_type=q_data.get("question_type"),
            category=q_data.get("category"),
            contexts=contexts,
            choices=q_data.get("choices"),
            answer=q_data.get("answer")
        )

    def _get_model_response(self, prompt: str) -> str:

        print("\n--- WYWOŁANIE MODELU (SYMULACJA) ---")
        print(f"Prompt (pierwsze 200 znaków): {prompt[:200]}...")
        # Właściwa implementacja:
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**inputs)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return response
        return f"To jest symulowana odpowiedź dla modelu {self.model_name}."

    def run_evaluation(self, dataset_filepath: str, query: Optional[Callable[[dict], bool]] = None):
        print(f"Uruchamianie ewaluacji dla modelu: {self.model_name}")
        self.dataset_manager.add_data_from_json_file(dataset_filepath)

        # Filtruj pytania zgodnie z podanym zapytaniem
        filtered_questions = self.dataset_manager._filter(self.dataset_manager.questions, query)
        print(f"Znaleziono {len(filtered_questions)} pytań do przetworzenia.")

        for q_id, q_data in filtered_questions.items():
            # 1. Stwórz obiekt Question
            question = self._create_question_object(q_id, q_data)

            # 2. Wygeneruj strukturę wiadomości dla promptu
            messages = self.prompt_manager.get_single_question_prompt(
                model_name=self.model_name,
                question_type=question.question_type,
                question_category=question.category,
                question_text=question.question_text,
                context_texts=[ctx.context_content for ctx in question.contexts],
                choices=question.choices
            )

            # 3. Zastosuj szablon czatu specyficzny dla modelu
            prompt_string = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 4. Uzyskaj odpowiedź z modelu (obecnie symulowane)
            response = self._get_model_response(prompt_string)

            # 5. Zapisz wynik
            self.results.append({
                "question_id": q_id,
                "model_name": self.model_name,
                "prompt_used": prompt_string,
                "model_response": response,
                "correct_answer": question.answer  # dla późniejszej oceny
            })
            print(f"Przetworzono pytanie: {q_id}")

    def save_results(self, output_filepath: str):
        """Zapisuje wyniki ewaluacji do pliku JSON."""
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\nWyniki zapisano w pliku: {output_filepath}")


if __name__ == '__main__':
    # Używamy ścieżki do modelu, który jest popularny i ma zdefiniowany chat_template
    # To pozwala na uruchomienie kodu bez błędów, nawet bez fizycznego modelu
    MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_NAME = "mistral-7b"

    # Inicjalizacja silnika
    engine = LLMQAEngine(model_name=MODEL_NAME, model_path=MODEL_PATH)

    # Przykładowe zapytanie: przetwarzaj tylko pytania otwarte (open_text) z kategorii 'matura_język_polski'
    example_query = lambda q: q.get("question_type") == "open_text" and q.get("category") == "matura_język_polski"

    # Uruchomienie ewaluacji na dostarczonym zbiorze danych
    # Zakładając, że mvp_dataset...json jest w tym samym folderze co skrypt
    engine.run_evaluation(
        dataset_filepath="mvp_dataset_2025-06-08T20-42-43Z.json",
        query=example_query
    )

    # Zapisanie wyników
    engine.save_results(output_filepath=f"results_{MODEL_NAME}.json")