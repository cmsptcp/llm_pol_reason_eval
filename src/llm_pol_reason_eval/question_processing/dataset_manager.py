# Finalna wersja pliku: dataset_manager.py
import json
from collections import Counter, defaultdict
from pathlib import Path
from sortedcontainers import SortedDict
from typing import List, Dict, Any, Optional, Callable, Iterator


class DatasetManager:
    def __init__(self):
        self.contexts: SortedDict[str, Dict] = SortedDict()
        self.questions: SortedDict[str, Dict] = SortedDict()

    def add_data_from_json_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ctx_id, ctx in data.get("contexts", {}).items():
            self.contexts[ctx_id] = ctx
        for q_id, q in data.get("questions", {}).items():
            self.questions[q_id] = q

    def get_few_shot_examples(self, question_type: str, num_examples: int) -> List[Dict[str, Any]]:
        examples = []
        for q_data in self.questions.values():
            if len(examples) >= num_examples:
                break
            if q_data.get("question_type") == question_type and q_data.get("answer"):
                example_contexts = {
                    cid: self.contexts[cid]
                    for cid in q_data.get("context_ids", []) if cid in self.contexts
                }
                examples.append({"question": q_data, "contexts": example_contexts})
        return examples

    def generate_question_batches(self, batch_size: int = 1, query: Optional[Callable[[dict], bool]] = None,
                                  by_question_category: bool = True, by_question_type: bool = True) -> Iterator[Dict[str, Any]]:
        filtered_questions = {k: v for k, v in self.questions.items() if (query is None or query(v))}

        grouped_by_criteria = defaultdict(list)
        for q_id, q_data in filtered_questions.items():
            key_parts = []
            if by_question_category:
                key_parts.append(q_data.get("category", "unknown"))
            if by_question_type:
                key_parts.append(q_data.get("question_type", "unknown"))
            key = tuple(key_parts) if key_parts else ("all",)
            grouped_by_criteria[key].append((q_id, q_data))

        for group_key, questions_in_group in sorted(grouped_by_criteria.items()):
            for i in range(0, len(questions_in_group), batch_size):
                batch_questions_tuples = questions_in_group[i: i + batch_size]
                batch_questions_dict = dict(batch_questions_tuples)

                context_ids_in_batch = {cid for _, q_data in batch_questions_tuples for cid in
                                        q_data.get("context_ids", [])}
                batch_contexts_dict = {cid: self.contexts[cid] for cid in context_ids_in_batch if cid in self.contexts}

                yield {"questions": batch_questions_dict, "contexts": batch_contexts_dict}

    def save_all_data_to_json_file(self, filepath: str):
        data = {"contexts": self.contexts, "questions": self.questions}
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> Dict[str, List[Any]]:
        category_counter = Counter(q.get("category") for q in self.questions.values() if q.get("category"))
        type_counter = Counter(q.get("question_type") for q in self.questions.values() if q.get("question_type"))
        return {"question_category_stats": category_counter.most_common(),
                "question_type_stats": type_counter.most_common()}

    def get_question_category_list(self) -> list[str]:
        return sorted(list(set(q.get("category") for q in self.questions.values() if q.get("category"))))

    def get_question_type_list(self) -> list[str]:
        return sorted(list(set(q.get("question_type") for q in self.questions.values() if q.get("question_type"))))