import json
import uuid
import os
import re
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

        project_root = Path(__file__).resolve().parents[2]
        templates_path = project_root / "llm_pol_reason_eval/prompts/templates"
        self.prompt_manager = PromptManager(templates_dir=templates_path)

        self.results: List[ModelAnswerData] = []
        self.logger = None

    # Są problemy z loggerem w środowisku Jupyter, więc używamy prostego loggera
    def _create_simple_logger(self, dataset_filepath: str, output_filepath: str):
        output_path = Path(output_filepath)
        dataset_path = Path(dataset_filepath)
        iso_time = datetime.now().isoformat(timespec="seconds").replace(":", "-")
        log_name = f"{dataset_path.stem}_{iso_time}.log"
        log_path = output_path.parent / log_name
        print(f"Tworzenie prostego loggera: {log_path.absolute()}")

        return SimpleFileLogger(log_path)

    def _parse_single_answer(self, raw_answer_chunk: str) -> str:
        match = re.search(r'<answer>(.*?)</answer>', raw_answer_chunk, re.DOTALL)
        return match.group(1).strip() if match else raw_answer_chunk.strip()

    def _parse_batched_response(self, response_text: str, question_ids: List[str]) -> Dict[str, str]:
        separator = "[[---KONIEC ODPOWIEDZI---]]"
        raw_chunks = [chunk.strip() for chunk in response_text.split(separator) if chunk.strip()]
        # Ostrzeżenie przez logger
        if hasattr(self, "logger") and len(raw_chunks) != len(question_ids):
            self.logger.warning(f"OSTRZEŻENIE: Liczba odp. ({len(raw_chunks)}) != liczby pytań ({len(question_ids)}).")
        elif len(raw_chunks) != len(question_ids):
            print(f"OSTRZEŻENIE: Liczba odp. ({len(raw_chunks)}) != liczby pytań ({len(question_ids)}).")
        parsed_answers = [self._parse_single_answer(chunk) for chunk in raw_chunks]
        return {q_id: ans for q_id, ans in zip(question_ids, parsed_answers)}

    def _save_partial_results(self, output_filepath: str):
        partial_path = Path(output_filepath).with_suffix('.partial.json')
        output_data = {"model_answers": {res["answer_id"]: res for res in self.results}}
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def generate_answers(self,
                         dataset_filepath: str,
                         output_filepath: str,
                         model_cfg: Dict,
                         prompt_composition: Dict,
                         batch_size: int = 10,
                         query: Optional[Callable[[dict], bool]] = None,
                         param_overrides: Optional[Dict[str, Any]] = None
                         ) -> List[ModelAnswerData]:

        logger = self._create_simple_logger(dataset_filepath, output_filepath)
        self.logger = logger
        logger.info("Logger uruchomiony")
        self.results = []
        self.dataset_manager.add_data_from_json_file(dataset_filepath)
        batch_generator = self.dataset_manager.get_grouped_question_batches(batch_size=batch_size, query=query)
        param_overrides = param_overrides or {}

        for i, batch_data in enumerate(batch_generator):
            metadata = batch_data['metadata']
            q_type = metadata.get('question_type')
            question_ids_in_batch = list(batch_data['questions'].keys())

            logger.info(f"--- Przetwarzanie batcha {i + 1} | Typ: {q_type}, Pytania: {len(question_ids_in_batch)} ---")

            messages = self.prompt_manager.get_question_prompt(model_cfg, batch_data, prompt_composition)

            final_params = param_overrides.get('default', {}).copy()
            if q_type and q_type in param_overrides.get('per_type', {}):
                final_params.update(param_overrides['per_type'][q_type])

            template_args = {'enable_thinking': final_params.pop('enable_thinking', False)}
            prompt_string = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_args)
            logger.info(f"Przygotowano prompt dla batcha {i + 1}: {prompt_string}")

            batched_response = self.inference_client.get_response(
                prompt_string, generation_params_override=final_params)

            logger.info(f"Otrzymano odpowiedź z modelu: {batched_response}")

            parsed_answers = self._parse_batched_response(batched_response, question_ids_in_batch)

            for q_id, answer_text in parsed_answers.items():
                self.results.append({
                    "answer_id": f"ans_{uuid.uuid4()}", "question_id": q_id, "answer_text": answer_text,
                    "generated_by": f"{self.model_name} ({self.model_path})",
                    "generation_date": datetime.now(timezone.utc).isoformat(),
                })

            self._save_partial_results(output_filepath)
            logger.info(f"Zapisano częściowe wyniki po batchu {i + 1}. Łącznie: {len(self.results)} odpowiedzi.")

        logger.info("Zakończono generowanie odpowiedzi.")
        self.save_final_results(output_filepath)
        return self.results

    def save_final_results(self, output_filepath: str):
        output_path = Path(output_filepath)
        partial_path = output_path.with_suffix('.partial.json')
        output_data = {"model_answers": {res["answer_id"]: res for res in self.results}}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        if hasattr(self, "logger"):
            self.logger.info(f"Wyniki finalne zapisano w pliku: {output_path}")
        else:
            print(f"\nWyniki finalne zapisano w pliku: {output_path}")
        if partial_path.exists():
            try:
                partial_path.unlink()
            except OSError as e:
                if hasattr(self, "logger"):
                    self.logger.warning(f"Błąd podczas usuwania pliku częściowego: {e}")
                else:
                    print(f"Błąd podczas usuwania pliku częściowego: {e}")


class SimpleFileLogger:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Utwórz plik i zapisz nagłówek
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"=== Log rozpoczęty {datetime.now().isoformat()} ===\n")

    def info(self, message):
        print(f"INFO: {message}")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} INFO: {message}\n")
            f.flush()
            os.fsync(f.fileno())

    def warning(self, message):
        print(f"WARNING: {message}")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} WARNING: {message}\n")
            f.flush()
            os.fsync(f.fileno())