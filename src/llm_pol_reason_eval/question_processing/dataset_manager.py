import json
from sortedcontainers import SortedDict
from typing import List, Dict, Any, Optional, Callable, Iterator
from collections import Counter, defaultdict

from llm_pol_reason_eval.data_models import ContextData, QuestionData

class DatasetManager:
    def __init__(self):
        self.contexts: SortedDict[str, ContextData] = SortedDict()
        self.questions: SortedDict[str, QuestionData] = SortedDict()

    def add_data_from_json_file(self, filepath: str, duplicate_strategy: str = "skip"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._add_data(data, duplicate_strategy)

    def add_data_from_json_string(self, json_str: str, duplicate_strategy: str = "skip"):
        data = json.loads(json_str)
        self._add_data(data, duplicate_strategy)

    def _add_data(self, data: dict, duplicate_strategy: str):
        for ctx_id, ctx in data.get("contexts", {}).items():
            if ctx_id not in self.contexts or duplicate_strategy == "replace":
                self.contexts[ctx_id] = ctx
            elif duplicate_strategy == "merge":
                self.contexts[ctx_id].update(ctx)
        for q_id, q in data.get("questions", {}).items():
            if q_id not in self.questions or duplicate_strategy == "replace":
                self.questions[q_id] = q
            elif duplicate_strategy == "merge":
                 self.questions[q_id].update(q)

    @staticmethod
    def _filter(items: SortedDict, query: Optional[Callable[[dict], bool]] = None) -> SortedDict:
        if not query:
            return items
        return SortedDict((k, v) for k, v in items.items() if query(v))

    def _get_contexts_for_questions(self, question_dict: Dict[str, QuestionData]) -> SortedDict[str, ContextData]:
        ctx_ids = set(cid for q_data in question_dict.values() for cid in q_data.get("context_ids", []))
        return SortedDict({cid: self.contexts[cid] for cid in sorted(list(ctx_ids)) if cid in self.contexts})

    def generate_question_batches(self, batch_size: int = 10, with_contexts: bool = True,
                                  query: Optional[Callable[[dict], bool]] = None,
                                  by_q_category: bool = True, by_q_type: bool = True) -> Iterator[Dict[str, Any]]:
        """ Generuje batche pytań, opcjonalnie grupując po kategorii i/lub typie pytania. """
        filtered_questions = self._filter(self.questions, query)

        # Definiuj sposób grupowania w zależności od parametrów
        grouped_by_criteria = defaultdict(list)
        for q_id, q_data in filtered_questions.items():
            key_parts = []
            if by_q_category:
                key_parts.append(q_data.get("category", "unknown"))
            if by_q_type:
                key_parts.append(q_data.get("question_type", "unknown"))

            key = tuple(key_parts) if key_parts else ("all",)
            grouped_by_criteria[key].append((q_id, q_data))

        for group_key, questions_in_group in grouped_by_criteria.items():
            for i in range(0, len(questions_in_group), batch_size):
                batch_questions_tuples = questions_in_group[i:i + batch_size]
                batch_q_data = SortedDict({q_id: q_data for q_id, q_data in batch_questions_tuples})

                # Metadata zależy od wybranych opcji grupowania
                metadata = {}
                if by_q_category and by_q_type:
                    metadata = {'category': group_key[0], 'question_type': group_key[1]}
                elif by_q_category:
                    metadata = {'category': group_key[0]}
                elif by_q_type:
                    metadata = {'question_type': group_key[0]}

                batch_data = {"questions": batch_q_data, "metadata": metadata}
                if with_contexts:
                    batch_data["contexts"] = self._get_contexts_for_questions(batch_q_data)
                yield batch_data

    def generate_question_batches_as_json_strings(self, batch_size: int = 10, with_contexts: bool = True,
                                                     query: Optional[Callable[[dict], bool]] = None,
                                                     by_q_category: bool = True, by_q_type: bool = True) -> List[str]:
        batch_generator = self.generate_question_batches(
            batch_size=batch_size,
            with_contexts=with_contexts,
            query=query,
            by_q_category=by_q_category,
            by_q_type=by_q_type
        )
        return [json.dumps(batch, ensure_ascii=False) for batch in batch_generator]

    def get_questions_as_list(self, batch_size: int = 10, with_contexts: bool = True,
                              query: Optional[Callable[[dict], bool]] = None,
                              by_q_category: bool = True, by_q_type: bool = True) -> List[Dict[str, Any]]:
        batch_generator = self.generate_question_batches(
            batch_size=batch_size,
            with_contexts=with_contexts,
            query=query,
            by_q_category=by_q_category,
            by_q_type=by_q_type
        )
        return list(batch_generator)

    def save_question_batches_as_jsonl_file(self, filepath: str, batch_size: int = 100,
                                           with_contexts: bool = True,
                                           query: Optional[Callable[[dict], bool]] = None,
                                           by_q_category: bool = True, by_q_type: bool = True):
        json_strings = self.generate_question_batches_as_json_strings(
            batch_size, with_contexts, query, by_q_category, by_q_type
        )
        with open(filepath, "w", encoding="utf-8") as f:
            for json_string in json_strings:
                f.write(json_string + "\n")

    def get_questions_as_json_string(self, with_contexts: bool = True, query: Optional[Callable[[dict], bool]] = None,
                                     human_readable: bool = True) -> str:
        """Zwraca pytania jako JSON string, z opcjonalnym filtrowaniem i kontekstami."""
        filtered_questions = self._filter(self.questions, query)
        data = {"questions": filtered_questions}
        if with_contexts:
            data["contexts"] = self._get_contexts_for_questions(filtered_questions)
        serializer_args = {'ensure_ascii': False, 'indent': 2} if human_readable else {'ensure_ascii': False, 'separators': (",", ":")}
        return json.dumps(data, **serializer_args)

    def save_all_data_to_json_file(self, filepath: str, human_readable: bool = True):
        """Zapisuje wszystkie konteksty i pytania do pliku JSON."""
        data = {"contexts": self.contexts, "questions": self.questions}
        serializer_args = {'ensure_ascii': False, 'indent': 2} if human_readable else {'ensure_ascii': False, 'separators': (",", ":")}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, **serializer_args)

    def remove_questions_by_query(self, query: Callable[[dict], bool]):
        to_remove = [k for k, v in self.questions.items() if query(v)]
        for k in to_remove:
            del self.questions[k]

    def remove_contexts_by_query(self, query: Callable[[dict], bool], force: bool = False):
        """
        Usuwa konteksty pasujące do podanego query.
        Domyślnie nie usuwa kontekstów, które są powiązane z pytaniami,
        """
        contexts_to_remove_ids = {cid for cid, c_data in self.contexts.items() if query(c_data)}
        if not contexts_to_remove_ids:
            return
        if not force:
            linked_q_ids = {q_id for q_id, q_data in self.questions.items() if any(cid in contexts_to_remove_ids for cid in q_data.get("context_ids", []))}
            if linked_q_ids:
                raise ValueError(
                    f"Nie można usunąć kontekstów, ponieważ są powiązane z pytaniami: {list(linked_q_ids)[:5]}... "
                    f"Użyj force=True, aby kontynuować."
                )
        for cid in contexts_to_remove_ids:
            if cid in self.contexts:
                del self.contexts[cid]

    def get_stats(self) -> Dict[str, List[Any]]:
        """Statystyki dotyczące kategorii i typów pytań."""
        category_counter = Counter(q.get("category") for q in self.questions.values() if q.get("category"))
        type_counter = Counter(q.get("question_type") for q in self.questions.values() if q.get("question_type"))
        return {
            "question_category_stats": category_counter.most_common(),
            "question_type_stats": type_counter.most_common(),
        }

    def get_question_category_list(self) -> list[str]:
        return sorted({q["category"] for q in self.questions.values() if "category" in q})

    def get_question_type_list(self) -> list[str]:
        return sorted({q["question_type"] for q in self.questions.values() if "question_type" in q})
