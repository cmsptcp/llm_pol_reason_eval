import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from transformers import AutoTokenizer
from llm_pol_reason_eval.data_models import ModelAnswerData
from llm_pol_reason_eval.question_processing.dataset_manager import DatasetManager
from llm_pol_reason_eval.prompts.prompt_manager import PromptManager
from llm_pol_reason_eval.qa_engine.inference_client import InferenceClient


class LLMQAEngine:
    def __init__(self, model_name: str, model_path: str, inference_client: InferenceClient):
        self.model_name = model_name
        self.model_path = model_path
        self.inference_client = inference_client
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.dataset_manager = DatasetManager()
        templates_path = Path(__file__).parent.parent / "prompts" / "templates"
        self.prompt_manager = PromptManager(templates_dir=templates_path)
        self.results: List[ModelAnswerData] = []

    def _parse_batched_response(self, response_text: str, question_ids: List[str]) -> Dict[str, str]:
        separator = "[[---KONIEC ODPOWIEDZI---]]"
        answers = [ans.strip() for ans in response_text.split(separator) if ans.strip()]
        if len(answers) != len(question_ids):
            print(
                f"OSTRZEŻENIE: Liczba odpowiedzi ({len(answers)}) nie zgadza się z liczbą pytań ({len(question_ids)}).")
        return {q_id: ans for q_id, ans in zip(question_ids, answers)}

    def generate_answers(self, dataset_filepath: str, batch_size: int = 10,
                         query: Optional[Callable[[dict], bool]] = None,
                         param_overrides: Optional[Dict[str, Any]] = None) -> List[ModelAnswerData]:
        self.results = []
        self.dataset_manager.add_data_from_json_file(dataset_filepath)
        batch_generator = self.dataset_manager.get_grouped_question_batches(batch_size=batch_size, query=query)
        param_overrides = param_overrides or {}
        per_type_params = param_overrides.get('per_type', {})

        for batch_data in batch_generator:
            metadata = batch_data['metadata']
            q_type = metadata['question_type']
            question_ids_in_batch = list(batch_data['questions'].keys())

            print(f"\n--- Przetwarzanie batcha | Typ: {q_type}, Pytania: {len(question_ids_in_batch)} ---")

            messages = self.prompt_manager.get_question_prompt(self.model_name, batch_data)

            template_args = {}
            if 'enable_thinking' in param_overrides.get('default', {}):
                template_args['enable_thinking'] = param_overrides['default']['enable_thinking']

            prompt_string = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_args
            )

            override_params = per_type_params.get(q_type)
            batched_response = self.inference_client.get_response(prompt_string,
                                                                  generation_params_override=override_params)
            parsed_answers = self._parse_batched_response(batched_response, question_ids_in_batch)

            for q_id, answer_text in parsed_answers.items():
                self.results.append({
                    "answer_id": f"ans_{uuid.uuid4()}",
                    "question_id": q_id,
                    "answer_text": answer_text,
                    "generated_by": f"{self.model_name} ({self.model_path})",
                    "generation_date": datetime.now(timezone.utc).isoformat(),
                })

        print("\nZakończono generowanie odpowiedzi.")
        return self.results

    def save_results(self, output_filepath: str):
        output_data = {"model_answers": {res["answer_id"]: res for res in self.results}}
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nWyniki ewaluacji zapisano w pliku: {output_filepath}")
