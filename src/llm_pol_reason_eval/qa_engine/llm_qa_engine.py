import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import re
import sys
import time

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
        templates_path = project_root / "src/llm_pol_reason_eval/prompts/templates"
        if not templates_path.is_dir():
            templates_path = project_root / "llm_pol_reason_eval/prompts/templates"
            if not templates_path.is_dir():
                raise FileNotFoundError(
                    f"Nie znaleziono katalogu szablonów w '{templates_path}' ani w alternatywnej ścieżce.")

        self.prompt_manager = PromptManager(templates_dir=templates_path)
        self.results: List[ModelAnswerData] = []
        self.prompts: Dict[str, str] = {}
        self.logger = None

    def generate_answers(self,
                         dataset_filepath: str,
                         output_filepath: str,
                         model_cfg: Dict,
                         prompt_composition: Dict,
                         batch_size: int = 1,
                         query: Optional[Callable[[dict], bool]] = None,
                         param_overrides: Optional[Dict[str, Any]] = None,
                         skip_questions: int = 0, max_questions: Optional[int] = None,
                         ) -> List[ModelAnswerData]:
        total_start_time = time.perf_counter()
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

        total_end_time = time.perf_counter()
        total_duration = total_end_time - total_start_time
        self.logger.info(f"=== ZAKOŃCZONO WSZYSTKIE ZADANIA ===")
        self.logger.info(f"Całkowity czas przetwarzania: {total_duration:.2f} sekund.")

        return self.results

    def _prepare_final_configs(self, model_cfg: Dict, prompt_composition: Dict, param_overrides: Dict,
                               q_type: Optional[str]) -> (Dict, Dict):
        """Centralna funkcja do łączenia wszystkich konfiguracji."""

        # 1. Połącz parametry generacji (zawsze zaczynaj od kopii, by nie modyfikować oryginału)
        final_gen_params = model_cfg.get('generation_params', {}).copy()
        final_gen_params.update(param_overrides.get('default', {}))
        if q_type and q_type in param_overrides.get('per_type', {}):
            final_gen_params.update(param_overrides['per_type'][q_type])

        # 2. Połącz parametry szablonu
        final_composition = prompt_composition.copy()
        final_composition['template_params'] = final_composition.get('template_params', {}).copy()

        # Przekaż wszystkie parametry generacji do szablonu, aby miał do nich dostęp
        final_composition['template_params'].update(final_gen_params)

        return final_gen_params, final_composition

    def _process_batches(self, batch_generator, model_cfg: Dict, prompt_composition: Dict,
                         param_overrides: Dict, output_filepath: str,
                         skip_questions: int = 0, max_questions: Optional[int] = None):
        total_questions_processed = 0
        questions_skipped = 0

        for i, batch_data in enumerate(batch_generator):
            questions_in_batch = list(batch_data.get('questions', {}).items())
            if not questions_in_batch:
                continue

            # Logika skip i max questions pozostaje bez zmian
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

            # Używamy ujednoliconej logiki dla obu przypadków
            batch_results = self._handle_inference(
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
                self._save_prompts(output_filepath)
                self.logger.info(
                    f"Zapisano częściowe wyniki po batchu {i + 1}. "
                    f"Przetworzono {len(batch_results)} pytań. Łącznie: {total_questions_processed}."
                )

            if max_questions is not None and total_questions_processed >= max_questions:
                self.logger.info(f"Osiągnięto limit {max_questions} pytań. Koniec przetwarzania.")
                break

        self.logger.info(f"Zakończono. Łącznie przetworzono {total_questions_processed} pytań.")

    def _handle_inference(self, batch_data: Dict, model_cfg: Dict, prompt_composition: Dict,
                          param_overrides: Dict, batch_index: int) -> List[ModelAnswerData]:
        """Ujednolicona metoda do obsługi inferencji dla batcha (dowolnego rozmiaru)."""
        questions_map = batch_data.get('questions', {})
        q_ids_in_order = list(questions_map.keys())

        all_prompts_chatml = []
        all_final_gen_params = []

        # Ta pętla jest potrzebna, bo każde pytanie może mieć inny `q_type` i inne parametry
        for q_id in q_ids_in_order:
            q_data = questions_map[q_id]
            q_type = q_data.get('question_type', 'N/A')
            contexts_for_q = {cid: batch_data['contexts'][cid] for cid in q_data.get("context_ids", []) if
                              cid in batch_data['contexts']}

            # Używamy nowej, scentralizowanej funkcji do przygotowania konfiguracji
            final_gen_params, final_composition = self._prepare_final_configs(model_cfg, prompt_composition,
                                                                              param_overrides, q_type)

            messages = self.prompt_manager.prepare_question_chatml_prompt(
                model_cfg, q_data, contexts_for_q, final_composition
            )

            all_prompts_chatml.append(messages)
            all_final_gen_params.append(final_gen_params)

        # Na potrzeby `apply_chat_template` potrzebujemy też argumentu `enable_thinking`
        tokenizer_args = {}
        if final_gen_params.get('enable_thinking') is not None:
            tokenizer_args['enable_thinking'] = final_gen_params['enable_thinking']

        final_prompts = [
            self.tokenizer.apply_chat_template(chatml, tokenize=False, add_generation_prompt=True, **tokenizer_args)
            for chatml in all_prompts_chatml
        ]

        # Zakładamy, że dla jednego batcha parametry generacji są takie same (bierzemy pierwsze)
        # To uproszczenie, które w Twoim obecnym przypadku jest prawdziwe.
        final_generation_params_for_batch = all_final_gen_params[0]

        start_time = time.perf_counter()
        raw_responses = self.inference_client.get_responses_with_batching(
            final_prompts,
            generation_params_override=final_generation_params_for_batch
        )
        end_time = time.perf_counter()

        total_duration = end_time - start_time
        duration_per_request = total_duration / len(final_prompts) if final_prompts else 0
        self.logger.info(f"Batch przetworzony w {total_duration:.2f}s ({duration_per_request:.2f}s na pytanie).")

        model_config_details = {
            "model_config": model_cfg,
            "prompt_composition": prompt_composition,
            "generation_parameters": final_generation_params_for_batch
        }

        batch_model_answers = []
        for i, q_id in enumerate(q_ids_in_order):
            self.prompts[q_id] = final_prompts[i]
            raw_response = raw_responses[i]
            parsed_answer = self._parse_single_answer(raw_response)
            self.logger.info(f"Sparsowana odpowiedź dla Q_ID {q_id}: {parsed_answer[:100].replace(chr(10), ' ')}...")

            batch_model_answers.append(ModelAnswerData(
                model_answer_id=f"ans_{uuid.uuid4()}",
                question_id=q_id,
                model_answer_raw_text=raw_response,
                model_answer_clean_text=parsed_answer,
                generated_by=f"{self.model_name} ({self.model_path})",
                generation_date=datetime.now(timezone.utc).isoformat(),
                model_configuration=model_config_details,
                generation_time=duration_per_request
            ))
        return batch_model_answers

    # Metody pomocnicze (bez zmian)
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
        # Spróbuj wyciągnąć pełną odpowiedź w znacznikach <answer>...</answer>
        match_ans = re.search(r'<answer>(.*?)</answer>', raw_answer_text, re.DOTALL | re.IGNORECASE)
        if match_ans:
            return match_ans.group(1).strip()

        # Jeżeli brak zamykającego </answer>, przetwarzaj wszystko po otwarciu <answer>
        match_open = re.search(r'<answer>(.*)', raw_answer_text, re.DOTALL | re.IGNORECASE)
        if match_open:
            return match_open.group(1).strip()

        # Obsłuż różne "tagi myślenia"
        thinking_tags = [r'</thinking>', r'</think>', r'</inner_monologue>']
        for tag in thinking_tags:
            match_think = re.search(tag + r'(.*)', raw_answer_text, re.DOTALL | re.IGNORECASE)
            if match_think:
                return match_think.group(1).strip()

        self.logger.warning(f"Nie odnaleziono znacznika <answer>; zwracam całość.")
        return raw_answer_text.strip()

    def _setup_logger_and_reset_results(self, dataset_filepath: str, output_filepath: str):
        log_prefix = Path(dataset_filepath).stem
        self.logger = self._create_simple_logger(log_prefix, output_filepath)
        self.results = []
        self.prompts = {}

    def _write_results_to_json(self, filepath: Path, results_list: List[ModelAnswerData]):
        output_data = {"model_answers": {res["model_answer_id"]: res for res in results_list}}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def _save_partial_results(self, output_filepath: str):
        self._write_results_to_json(Path(output_filepath).with_suffix('.partial.json'), self.results)

    def _save_prompts(self, output_filepath: str):
        prompts_filepath = Path(output_filepath).with_name(f"{Path(output_filepath).stem}.prompts.json")
        output_data = {"prompts": self.prompts}
        prompts_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(prompts_filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

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
        print(log_entry.strip(), file=sys.stderr)
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                f.flush()
        except Exception as e:
            print(f"KRYTYCZNY BŁĄD LOGOWANIA do {self.filepath}: {e}", file=sys.stderr)

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

    def error(self, message: str):
        self._log("ERROR", message)