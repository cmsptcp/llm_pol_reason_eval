import json
from typing import List, Dict, Any, Optional, Callable

class DatasetManager:
    def __init__(self):
        self.contexts: Dict[str, dict] = {}
        self.questions: Dict[str, dict] = {}

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

    def _filter(self, items: Dict[str, dict], query: Optional[Callable[[dict], bool]] = None):
        if not query:
            return items
        return {k: v for k, v in items.items() if query(v)}

    def _sort(self, items: Dict[str, dict], sort_key: Optional[str] = None):
        if not sort_key:
            return items
        return dict(sorted(items.items(), key=lambda x: x[1].get(sort_key, "")))

    def get_questions_in_batches_as_jsonl_string(self, batch_size: int = 10, with_contexts: bool = False, query: Optional[Callable[[dict], bool]] = None, sort_key: Optional[str] = None) -> List[str]:
        questions = self._filter(self.questions, query)
        questions = self._sort(questions, sort_key)
        keys = list(questions.keys())
        batches = []
        for i in range(0, len(keys), batch_size):
            batch = {k: questions[k] for k in keys[i:i+batch_size]}
            if with_contexts:
                ctx_ids = set()
                for q in batch.values():
                    ctx_ids.update(q.get("context_ids", []))
                batch_contexts = {cid: self.contexts[cid] for cid in ctx_ids if cid in self.contexts}
                batch_data = {"questions": batch, "contexts": batch_contexts}
            else:
                batch_data = {"questions": batch}
            batches.append(json.dumps(batch_data, ensure_ascii=False))
        return batches

    def save_questions_in_batches_as_jsonl_file(self, filepath: str, batch_size: int = 100, with_contexts: bool = False, query: Optional[Callable[[dict], bool]] = None, sort_key: Optional[str] = None):
        batches = self.get_questions_in_batches_as_jsonl_string(batch_size, with_contexts, query, sort_key)
        with open(filepath, "w", encoding="utf-8") as f:
            for batch in batches:
                f.write(batch + "\n")

    def get_contexts_as_json_string(self, query: Optional[Callable[[dict], bool]] = None, sort_key: Optional[str] = None) -> str:
        contexts = self._filter(self.contexts, query)
        contexts = self._sort(contexts, sort_key)
        return json.dumps(contexts, ensure_ascii=False)

    def get_questions_as_json_string(self, with_contexts: bool = False, query: Optional[Callable[[dict], bool]] = None, sort_key: Optional[str] = None) -> str:
        questions = self._filter(self.questions, query)
        questions = self._sort(questions, sort_key)
        if with_contexts:
            ctx_ids = set()
            for q in questions.values():
                ctx_ids.update(q.get("context_ids", []))
            batch_contexts = {cid: self.contexts[cid] for cid in ctx_ids if cid in self.contexts}
            return json.dumps({"questions": questions, "contexts": batch_contexts}, ensure_ascii=False)
        return json.dumps({"questions": questions}, ensure_ascii=False)

    def get_all_data_as_json_string(self) -> str:
        return json.dumps({"contexts": self.contexts, "questions": self.questions}, ensure_ascii=False)

    def save_all_data_to_json_file(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.get_all_data_as_json_string())

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
        self.contexts = {
            cid: context_data
            for cid, context_data in self.contexts.items()
            if cid not in contexts_to_remove_ids
        }

    def remove_questions_by_query(self, query: Callable[[dict], bool]):
        to_remove = [k for k, v in self.questions.items() if query(v)]
        for k in to_remove:
            del self.questions[k]