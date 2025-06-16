import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import re

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

        if not templates_path.is_dir():
            alt_templates_path = project_root / "src/llm_pol_reason_eval/prompts/templates"
            if not alt_templates_path.is_dir():
                raise FileNotFoundError(
                    f"Nie znaleziono katalogu szablonów w '{templates_path}' ani '{alt_templates_path}'")
            templates_path = alt_templates_path

        self.prompt_manager = PromptManager(templates_dir=templates_path)
        self.results: List[ModelAnswerData] = []
        self.logger = None

    def generate_answers(self,
                         dataset_filepath: str,
                         output_filepath: str,
                         model_cfg: Dict,
                         prompt_composition: Dict,
                         batch_size: int = 10,
                         query: Optional[Callable[[dict], bool]] = None,
                         param_overrides: Optional[Dict[str, Any]] = None,
                         skip_questions: int = 0, max_questions: Optional[int] = None,
                         ) -> List[ModelAnswerData]:
        self._setup_logger_and_reset_results(dataset_filepath, output_filepath)
        self._load_dataset(dataset_filepath)
        batch_generator = self._create_question_batch_iterator(batch_size, query)
        self._process_batches(
            batch_generator,
            model_cfg,
            prompt_composition,
            param_overrides or {},
            output_filepath,
            skip_questions,
            max_questions
        )
        self.save_final_results(output_filepath)
        return self.results

    def _process_batches(self, batch_generator, model_cfg: Dict, prompt_composition: Dict,
                         param_overrides: Dict, output_filepath: str,
                         skip_questions: int = 0, max_questions: Optional[int] = None):
        total_questions_processed = 0
        questions_skipped = 0

        for i, batch_data in enumerate(batch_generator):
            questions_in_batch = list(batch_data.get('questions', {}).items())
            if not questions_in_batch:
                continue

            if questions_skipped < skip_questions:
                to_skip = min(skip_questions - questions_skipped, len(questions_in_batch))
                questions_in_batch = questions_in_batch[to_skip:]
                questions_skipped += to_skip
                if not questions_in_batch:
                    continue

            if max_questions is not None and total_questions_processed + len(questions_in_batch) > max_questions:
                questions_in_batch = questions_in_batch[:max_questions - total_questions_processed]

            if not questions_in_batch:
                continue

            batch_data['questions'] = dict(questions_in_batch)
            self.logger.info(f"Przetwarzanie batcha {i + 1} z {len(questions_in_batch)} pytaniami.")

            # Logika dyspozytorska: wybór trybu przetwarzania
            if len(questions_in_batch) > 1:
                batch_results = self._handle_batched_inference(
                    batch_data=batch_data,
                    model_cfg=model_cfg,
                    prompt_composition=prompt_composition,
                    param_overrides=param_overrides,
                    batch_index=i
                )
            else:
                batch_results = self._handle_serial_processing(
                    batch_data=batch_data,
                    model_cfg=model_cfg,
                    prompt_composition=prompt_composition,
                    param_overrides=param_overrides,
                    batch_index=i
                )

            if batch_results:
                self.results.extend(batch_results)
                total_questions_processed += len(batch_results)
                self._save_partial_results(output_filepath)
                self.logger.info(
                    f"Zapisano częściowe wyniki po batchu {i + 1}. "
                    f"Przetworzono {len(batch_results)} pytań. Łącznie: {total_questions_processed}."
                )

            if max_questions is not None and total_questions_processed >= max_questions:
                self.logger.info(f"Osiągnięto limit {max_questions} pytań. Koniec przetwarzania.")
                break

        self.logger.info(f"Zakończono. Łącznie przetworzono {total_questions_processed} pytań.")

    def _handle_batched_inference(self, batch_data: Dict, model_cfg: Dict, prompt_composition: Dict,
                                  param_overrides: Dict, batch_index: int) -> List[ModelAnswerData]:
        """Przetwarza cały batch pytań za jednym razem, wykorzystując `get_responses_with_batching`."""
        self.logger.info(f"--- Batch {batch_index + 1}: tryb wsadowy ---")

        questions_map = batch_data.get('questions', {})
        q_ids_in_order = list(questions_map.keys())

        all_chatml_prompts = self.prompt_manager.prepare_question_chatml_prompt_batch(
            model_cfg, batch_data, prompt_composition
        )
        final_prompts = [
            self.tokenizer.apply_chat_template(chatml, tokenize=False, add_generation_prompt=True)
            for chatml in all_chatml_prompts
        ]

        q_type = list(questions_map.values())[0].get('question_type', 'N/A')
        generation_params, _ = self._get_generation_params_and_tokenizer_args(q_type, param_overrides)

        self.logger.info(f"Wysyłanie {len(final_prompts)} promptów do modelu z parametrami: {generation_params}")
        raw_responses = self.inference_client.get_responses_with_batching(final_prompts, generation_params)

        model_config_details_json = json.dumps({
            "model_config": model_cfg, "prompt_composition": prompt_composition,
            "generation_parameters": generation_params
        }, ensure_ascii=False)

        batch_model_answers = []
        for i, q_id in enumerate(q_ids_in_order):
            raw_response = raw_responses[i]
            parsed_answer = self._parse_single_answer(raw_response)
            self.logger.info(f"Sparsowana odpowiedź dla Q_ID {q_id}: {parsed_answer[:100]}...")
            batch_model_answers.append(ModelAnswerData(
                model_answer_id=f"ans_{uuid.uuid4()}",
                question_id=q_id,
                model_answer_raw_text=raw_response,
                model_answer_clean_text=parsed_answer,
                generated_by=f"{self.model_name} ({self.model_path})",
                generation_date=datetime.now(timezone.utc).isoformat(),
                model_configuration=model_config_details_json
            ))
        return batch_model_answers

    def _handle_serial_processing(self, batch_data: Dict, model_cfg: Dict, prompt_composition: Dict,
                                  param_overrides: Dict, batch_index: int) -> List[ModelAnswerData]:
        """Przetwarza pytania jedno po drugim (dla batch_size=1)."""
        self.logger.info(f"--- Batch {batch_index + 1}: tryb seryjny ---")

        q_id, q_data = list(batch_data.get('questions', {}).items())[0]
        all_contexts = batch_data.get('contexts', {})
        contexts_for_q = {cid: all_contexts[cid] for cid in q_data.get("context_ids", []) if cid in all_contexts}

        messages = self.prompt_manager.prepare_question_chatml_prompt(
            model_cfg, q_data, contexts_for_q, prompt_composition
        )

        q_type = q_data.get('question_type', 'N/A')
        generation_params, tokenizer_args = self._get_generation_params_and_tokenizer_args(q_type, param_overrides)

        prompt_string = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **tokenizer_args
        )

        raw_response = self.inference_client.get_response(prompt_string, generation_params)
        parsed_answer = self._parse_single_answer(raw_response)
        self.logger.info(f"Sparsowana odpowiedź dla Q_ID {q_id}: {parsed_answer[:100]}...")

        model_config_details = {
            "model_config": model_cfg, "prompt_composition": prompt_composition,
            "generation_parameters": generation_params, "tokenizer_arguments": tokenizer_args
        }

        return [ModelAnswerData(
            model_answer_id=f"ans_{uuid.uuid4()}",
            question_id=q_id,
            model_answer_raw_text=raw_response,
            model_answer_clean_text=parsed_answer,
            generated_by=f"{self.model_name} ({self.model_path})",
            generation_date=datetime.now(timezone.utc).isoformat(),
            model_configuration=json.dumps(model_config_details, ensure_ascii=False)
        )]

    def _get_generation_params_and_tokenizer_args(self, q_type: Optional[str], param_overrides: Dict) -> (Dict, Dict):
        final_gen_params = param_overrides.get('default', {}).copy()
        if q_type and q_type in param_overrides.get('per_type', {}):
            final_gen_params.update(param_overrides['per_type'][q_type])
        tokenizer_args = {'enable_thinking': final_gen_params.pop('enable_thinking', False)}
        return final_gen_params, tokenizer_args

    def _load_dataset(self, dataset_filepath: str):
        self.logger.info(f"Ładowanie datasetu z: {dataset_filepath}")
        self.dataset_manager.add_data_from_json_file(dataset_filepath)
        self.logger.info(f"Załadowano {len(self.dataset_manager.questions)} pytań.")

    def _create_question_batch_iterator(self, batch_size: int, query: Optional[Callable[[dict], bool]] = None):
        self.logger.info(f"Tworzenie iteratora z rozmiarem batcha: {batch_size}.")
        return self.dataset_manager.generate_question_batches(
            batch_size=batch_size, query=query, by_q_category=True, by_q_type=True, with_contexts=True
        )

    def _parse_single_answer(self, raw_answer_text: str) -> str:
        match = re.search(r'<answer>(.*?)</answer>', raw_answer_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        self.logger.warning(f"Nie znaleziono tagu <answer> w odpowiedzi: '{raw_answer_text[:100]}...'")
        return raw_answer_text.strip()

    def _setup_logger_and_reset_results(self, dataset_filepath: str, output_filepath: str):
        log_prefix = Path(dataset_filepath).stem
        self.logger = self._create_simple_logger(log_prefix, output_filepath)
        self.logger.info("Logger uruchomiony.")
        self.results = []

    def _write_results_to_json(self, filepath: Path, results_list: List[ModelAnswerData]):
        output_data = {"model_answers": {res["model_answer_id"]: res for res in results_list}}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def _save_partial_results(self, output_filepath: str):
        self._write_results_to_json(Path(output_filepath).with_suffix('.partial.json'), self.results)

    def save_final_results(self, output_filepath: str):
        output_path = Path(output_filepath)
        self._write_results_to_json(output_path, self.results)
        self.logger.info(f"Wyniki finalne ({len(self.results)} odpowiedzi) zapisano w: {output_path}")

        partial_path = output_path.with_suffix('.partial.json')
        if partial_path.exists():
            try:
                partial_path.unlink()
                self.logger.info(f"Usunięto plik częściowy: {partial_path}")
            except OSError as e:
                self.logger.warning(f"Błąd podczas usuwania pliku częściowego '{partial_path}': {e}")

    def _create_simple_logger(self, log_prefix: str, output_filepath: str):
        output_path = Path(output_filepath)
        iso_time = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-").replace("+00-00", "Z")
        log_name = f"{log_prefix}_{self.model_name.replace('/', '_')}_{iso_time}.log"
        log_dir = output_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
        print(f"Ścieżka logów: {log_path.resolve()}")
        return SimpleFileLogger(log_path)


class SimpleFileLogger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"=== Log rozpoczęty {datetime.now(timezone.utc).isoformat()} ===\n")
            f.flush()

    def _log(self, level: str, message: str):
        log_entry = f"{datetime.now(timezone.utc).isoformat()} {level}: {message}\n"
        print(log_entry.strip())
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                f.flush()
        except Exception as e:
            print(f"KRYTYCZNY BŁĄD LOGOWANIA do {self.filepath}: {e}")

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

    def error(self, message: str):
        self._log("ERROR", message)