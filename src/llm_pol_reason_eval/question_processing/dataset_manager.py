import json
from sortedcontainers import SortedDict
from typing import List, Dict, Any, Optional, Callable
from collections import Counter


class DatasetManager:
    def __init__(self):
        self.contexts: SortedDict[str, dict] = SortedDict()
        self.questions: SortedDict[str, dict] = SortedDict()

    def add_data_from_json_file(self, filepath: str, duplicate_strategy: str = "skip"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._add_data(data, duplicate_strategy)

    def add_data_from_json_string(self, json_str: str, duplicate_strategy: str = "skip"):
        data = json.loads(json_str)
        self._add_data(data, duplicate_strategy)

    def _add_data(self, data: dict, duplicate_strategy: str):
        for ctx_id, ctx in data.get("contexts", {}).items():
            if ctx_id in self.contexts:
                if duplicate_strategy == "replace":
                    self.contexts[ctx_id] = ctx
                elif duplicate_strategy == "merge":
                    self.contexts[ctx_id].update(ctx)
            else:
                self.contexts[ctx_id] = ctx

        for q_id, q in data.get("questions", {}).items():
            if q_id in self.questions:
                if duplicate_strategy == "replace":
                    self.questions[q_id] = q
                elif duplicate_strategy == "merge":
                    self.questions[q_id].update(q)
            else:
                self.questions[q_id] = q

    @staticmethod
    def _filter(items: SortedDict[str, dict], query: Optional[Callable[[dict], bool]] = None) -> SortedDict[str, dict]:
        if not query:
            return items
        return SortedDict((k, v) for k, v in items.items() if query(v))

    def get_questions_in_batches_as_jsonl_string(self, batch_size: int = 10, with_contexts: bool = True,
                                                 query: Optional[Callable[[dict], bool]] = None) -> List[str]:
        """
        Generates batches of questions as JSONL string.
        Each batch (line of JSONL string) is a JSON object containing questions and optionally contexts.
        """
        filtered_questions = self._filter(self.questions, query)
        keys = list(filtered_questions.keys())
        batches = []
        for i in range(0, len(keys), batch_size):
            batch_q_data = SortedDict()
            current_keys = keys[i:i + batch_size]
            for k in current_keys:
                batch_q_data[k] = filtered_questions[k]

            batch_data: Dict[str, Any]
            if with_contexts:
                ctx_ids = set()
                for q in batch_q_data.values():
                    ctx_ids.update(q.get("context_ids", []))

                batch_c_data = SortedDict()
                for cid in ctx_ids:  # SortedDict posortuje klucze automatycznie
                    if cid in self.contexts:
                        batch_c_data[cid] = self.contexts[cid]
                batch_data = {"questions": batch_q_data, "contexts": batch_c_data}
            else:
                batch_data = {"questions": batch_q_data}
            batches.append(json.dumps(batch_data, ensure_ascii=False))
        return batches

    def save_questions_in_batches_as_jsonl_file(self, filepath: str, batch_size: int = 100, with_contexts: bool = True,
                                                query: Optional[Callable[[dict], bool]] = None):
        batches = self.get_questions_in_batches_as_jsonl_string(batch_size, with_contexts, query)
        with open(filepath, "w", encoding="utf-8") as f:
            for batch in batches:
                f.write(batch + "\n")

    def get_contexts_as_json_string(self, query: Optional[Callable[[dict], bool]] = None,
                                    human_readable: bool = True) -> str:
        filtered_contexts = self._filter(self.contexts, query)
        if human_readable:
            return json.dumps(filtered_contexts, ensure_ascii=False, indent=2, sort_keys=False)
        return json.dumps(filtered_contexts, ensure_ascii=False, separators=(",", ":"))

    def get_questions_as_json_string(self, with_contexts: bool = True, query: Optional[Callable[[dict], bool]] = None,
                                     human_readable: bool = True) -> str:
        filtered_questions = self._filter(self.questions, query)

        data: Dict[str, Any]
        if with_contexts:
            ctx_ids = set()
            for q in filtered_questions.values():
                ctx_ids.update(q.get("context_ids", []))

            contexts_data = SortedDict()
            for cid in ctx_ids:  # SortedDict posortuje klucze automatycznie
                if cid in self.contexts:
                    contexts_data[cid] = self.contexts[cid]
            data = {"questions": filtered_questions, "contexts": contexts_data}
        else:
            data = {"questions": filtered_questions}

        if human_readable:
            return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def get_all_data_as_json_string(self, human_readable: bool = True) -> str:
        data = {"contexts": self.contexts, "questions": self.questions}
        if human_readable:
            return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def save_all_data_to_json_file(self, filepath: str, human_readable: bool = True):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.get_all_data_as_json_string(human_readable=human_readable))

    def remove_contexts_by_query(self, query: Callable[[dict], bool], force: bool = False):
        contexts_to_remove_ids = {
            cid for cid, context_data in self.contexts.items() if query(context_data)
        }

        if not contexts_to_remove_ids:
            return

        if not force:
            for question_data in self.questions.values():
                if "context_ids" in question_data:
                    for linked_ctx_id in question_data["context_ids"]:
                        if linked_ctx_id in contexts_to_remove_ids:
                            raise ValueError(
                                f"Kontekst {linked_ctx_id} jest powiązany z co najmniej jednym pytaniem i nie może "
                                f"zostać usunięty bez użycia opcji force=True."
                            )

        new_contexts = SortedDict()
        for cid, context_data in self.contexts.items():
            if cid not in contexts_to_remove_ids:
                new_contexts[cid] = context_data
        self.contexts = new_contexts

    def remove_questions_by_query(self, query: Callable[[dict], bool]):
        to_remove = [k for k, v in self.questions.items() if query(v)]
        for k in to_remove:
            del self.questions[k]

    def get_stats(self) -> Dict[str, List[Any]]:
        category_counter = Counter()
        type_counter = Counter()
        category_type_counter = Counter()

        for question_data in self.questions.values():
            category = question_data.get("category")
            question_type = question_data.get("question_type")

            if category:
                category_counter[category] += 1
            if question_type:
                type_counter[question_type] += 1
            if category and question_type:
                category_type_counter[(category, question_type)] += 1

        question_category_stats = category_counter.most_common()
        question_type_stats = type_counter.most_common()
        question_category_by_type_stats = category_type_counter.most_common()

        return {
            "question_category_stats": question_category_stats,
            "question_type_stats": question_type_stats,
            "question_category_by_type_stats": question_category_by_type_stats,
        }

    def get_question_category_list(self) -> list[str]:
        categories = set()
        for question_data in self.questions.values():
            if "category" in question_data:
                categories.add(question_data["category"])
        return sorted(list(categories))

    def get_question_type_list(self) -> list[str]:
        question_types = set()
        for question_data in self.questions.values():
            if "question_type" in question_data:
                question_types.add(question_data["question_type"])
        return sorted(list(question_types))
