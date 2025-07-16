import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import re
import sys
import time

from llm_pol_reason_eval.data_models import ModelAnswerData
from llm_pol_reason_eval.question_processing.dataset_manager import DatasetManager
from llm_pol_reason_eval.prompts.prompt_manager import PromptManager
from llm_pol_reason_eval.qa_engine.inference_client import InferenceClient


class LLMQAEngine:
    def __init__(self, model_name: str, inference_client: InferenceClient, project_root: Optional[Path] = None):
        self.model_name = model_name
        self.inference_client = inference_client
        self.tokenizer = self.inference_client.tokenizer
        self.dataset_manager = DatasetManager()
        self.examples_manager = DatasetManager()

        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        if project_root is None:
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
        self.prompts: Dict[str, str] = {}
        self.logger = None

    def generate_answers(self,
                         dataset_filepath: str, output_filepath: str, model_cfg: Dict,
                         prompt_composition: Dict, batch_size: int = 1,
                         examples_dataset_filepath: Optional[str] = None,
                         query: Optional[Callable[[dict], bool]] = None,
                         param_overrides: Optional[Dict[str, Any]] = None,
                         skip_questions: int = 0, max_questions: Optional[int] = None) -> List[ModelAnswerData]:

        total_start_time = time.perf_counter()
        self._setup_logger_and_reset_results(dataset_filepath, output_filepath)

        self.dataset_manager.add_data_from_json_file(dataset_filepath)
        if "few_shot" in prompt_composition.get("components", []) and examples_dataset_filepath:
            self.logger.info(f"Ładowanie dedykowanego zbioru przykładów z: {examples_dataset_filepath}")
            self.examples_manager.add_data_from_json_file(examples_dataset_filepath)

        batch_generator = self.dataset_manager.generate_question_batches(batch_size=batch_size, query=query)
        self._process_batches(
            batch_generator, model_cfg, prompt_composition, param_overrides or {}, output_filepath,
            skip_questions, max_questions
        )

        self.save_final_results(output_filepath)
        total_end_time = time.perf_counter()
        total_duration = total_end_time - total_start_time
        self.logger.info(f"=== ZAKOŃCZONO WSZYSTKIE ZADANIA ===")
        self.logger.info(f"Całkowity czas przetwarzania: {total_duration:.2f} sekund.")
        return self.results

    def _prepare_final_configs(self, model_cfg: Dict, prompt_composition: Dict,
                               param_overrides: Dict, q_type: Optional[str]) -> (Dict, Dict):
        final_gen_params = model_cfg.get('generation_params', {}).copy()
        final_gen_params.update(param_overrides.get('default', {}))
        if q_type and q_type in param_overrides.get('per_type', {}):
            final_gen_params.update(param_overrides['per_type'][q_type])

        final_composition = prompt_composition.copy()
        final_composition['template_params'] = final_composition.get('template_params', {}).copy()
        final_composition['template_params'].update(final_gen_params)
        return final_gen_params, final_composition

    def _process_batches(self, batch_generator, model_cfg: Dict, prompt_composition: Dict,
                         param_overrides: Dict, output_filepath: str,
                         skip_questions: int = 0, max_questions: Optional[int] = None):
        total_questions_processed = 0
        for i, batch_data in enumerate(batch_generator):
            if not batch_data.get('questions'): continue

            self.logger.info(f"Przetwarzanie batcha {i + 1} z {len(batch_data['questions'])} pytaniami.")

            batch_results = self._handle_inference(
                batch_data, model_cfg, prompt_composition, param_overrides
            )
            if batch_results:
                self.results.extend(batch_results)
                total_questions_processed += len(batch_results)
                self._save_partial_results(output_filepath)
                self._save_prompts(output_filepath)

            if max_questions is not None and total_questions_processed >= max_questions:
                self.logger.info(f"Osiągnięto limit {max_questions} pytań. Koniec przetwarzania.")
                break
        self.logger.info(f"Zakończono. Łącznie przetworzono {total_questions_processed} pytań.")

    def _handle_inference(self, batch_data: Dict, model_cfg: Dict, prompt_composition: Dict,
                          param_overrides: Dict) -> List[ModelAnswerData]:
        q_ids_in_order = list(batch_data.get('questions', {}).keys())

        prompts_for_batch = []
        final_gen_params_for_batch = {}

        for q_id in q_ids_in_order:
            q_data = batch_data['questions'][q_id]
            q_type = q_data.get('question_type', 'N/A')
            contexts_for_q = {cid: batch_data['contexts'][cid] for cid in q_data.get("context_ids", []) if
                              cid in batch_data['contexts']}

            final_gen_params, final_composition = self._prepare_final_configs(model_cfg, prompt_composition,
                                                                              param_overrides, q_type)

            few_shot_examples = []
            if "few_shot" in final_composition.get("components", []):
                num_examples = final_composition.get('template_params', {}).get('num_few_shot', 2)
                few_shot_examples = self.examples_manager.get_few_shot_examples(
                    question_type=q_type,
                    num_examples=num_examples
                )

            messages = self.prompt_manager.prepare_conversation_prompt(
                model_cfg, q_data, contexts_for_q, final_composition, few_shot_examples
            )

            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts_for_batch.append(prompt_str)
            self.prompts[q_id] = str(messages)
            final_gen_params_for_batch = final_gen_params

        if not prompts_for_batch: return []

        raw_responses = self.inference_client.get_responses_with_batching(
            prompts_for_batch,
            generation_params_override=final_gen_params_for_batch
        )

        duration_per_request = len(raw_responses) / len(q_ids_in_order) if q_ids_in_order else 0

        batch_model_answers = []
        for i, q_id in enumerate(q_ids_in_order):
            batch_model_answers.append(ModelAnswerData(
                model_answer_id=f"ans_{uuid.uuid4()}", question_id=q_id, model_answer_raw_text=raw_responses[i],
                model_answer_clean_text=self._parse_single_answer(raw_responses[i]),
                generated_by=f"{self.model_name} ({model_cfg.get('path')})",
                generation_date=datetime.now(timezone.utc).isoformat(),
                model_configuration={"model_config": model_cfg, "prompt_composition": prompt_composition,
                                     "generation_parameters": final_gen_params_for_batch},
                generation_time=duration_per_request
            ))
        return batch_model_answers

    def _parse_single_answer(self, raw_answer_text: str) -> str:
        matches = re.findall(r'<answer>(.*?)</answer>', raw_answer_text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        open_tag_match = re.search(r'<answer>(.*)', raw_answer_text, re.DOTALL | re.IGNORECASE)
        if open_tag_match:
            return open_tag_match.group(1).strip()

        thinking_tags = [r'</thinking>', r'</think>', r'</inner_monologue>']
        last_content = None
        last_index = -1

        for tag in thinking_tags:
            for m in re.finditer(tag + r'(.*)', raw_answer_text, re.DOTALL | re.IGNORECASE):
                if m.start() > last_index:
                    last_index = m.start()
                    last_content = m.group(1)

        return last_content.strip() if last_content else raw_answer_text.strip()

    def _setup_logger_and_reset_results(self, dataset_filepath: str, output_filepath: str):
        log_prefix = Path(dataset_filepath).stem
        self.logger = self._create_simple_logger(log_prefix, output_filepath)
        self.results = []
        self.prompts = {}

    def _create_simple_logger(self, log_prefix: str, output_filepath: str):
        output_path = Path(output_filepath)
        iso_time = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-").replace("+00-00", "Z")
        log_name = f"{log_prefix}_{self.model_name.replace('/', '_')}_{iso_time}.log"
        log_dir = output_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
        print(f"Ścieżka logów: {log_path.resolve()}")
        return SimpleFileLogger(log_path)

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


class SimpleFileLogger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'a', encoding='utf-8') as f: f.write(
            f"=== Log rozpoczęty {datetime.now(timezone.utc).isoformat()} ===\n")

    def _log(self, level: str, message: str):
        log_entry = f"{datetime.now(timezone.utc).isoformat()} {level}: {message}\n"
        print(log_entry.strip(), file=sys.stderr)
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"KRYTYCZNY BŁĄD LOGOWANIA do {self.filepath}: {e}", file=sys.stderr)

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

    def error(self, message: str):
        self._log("ERROR", message)