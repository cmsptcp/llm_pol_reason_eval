import pandas as pd
import json
from enum import Enum
from typing import List, Dict, Optional, Any
from llm_pol_reason_eval.utils.logger import get_logger

logger = get_logger(__name__)


class DuplicatesStrategy(Enum):
    REPLACE_WITH_NEW = "replace_with_new"
    IGNORE_DUPLICATES = "ignore_duplicates"


class DatasetProcessor:
    def __init__(self):
        self.contexts_df = pd.DataFrame()
        self.questions_df = pd.DataFrame()
        logger.info("DatasetProcessor zainicjalizowany.")

    def _update_dataframe(self, attr: str, new_data: List[Dict], id_column: str,
                          dup_strategy: DuplicatesStrategy):
        if not new_data:
            logger.debug(f"Brak nowych danych do dodania do {attr}.")
            return
        existing = getattr(self, attr)
        new_df = pd.DataFrame(new_data)
        if new_df.empty:
            logger.debug(f"Nowe dane do {attr} są puste.")
            return
        if dup_strategy == DuplicatesStrategy.REPLACE_WITH_NEW:
            combined = pd.concat([new_df, existing], ignore_index=True)
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)
        if id_column in combined.columns:
            combined.drop_duplicates(subset=[id_column], keep='first', inplace=True)
        elif not combined.empty:
            logger.warning(f"Brak kolumny ID '{id_column}' w {attr}.")
        setattr(self, attr, combined)
        logger.debug(f"Zaktualizowano {attr}. Liczba rekordów: {len(getattr(self, attr))}")

    def _sort_list(self, data: List[Dict], key: Optional[str]) -> List[Dict]:
        if key and data:
            data.sort(key=lambda x: str(x.get(key, '')))
        return data

    def _serialize_df(self, df: pd.DataFrame, key: Optional[str]) -> List[Dict]:
        if df.empty:
            return []
        return self._sort_list(df.to_dict(orient='records'), key)

    def _get_data_as_dict(self, questions_df_override: Optional[pd.DataFrame] = None, include_contexts: bool = True,
                           include_questions: bool = True, sort_contexts_by_key: Optional[str] = None,
                           sort_questions_by_key: Optional[str] = None) -> Dict[str, List[Dict]]:
        result = {}
        questions_df = questions_df_override if questions_df_override is not None else self.questions_df
        if include_contexts:
            contexts_df = self.contexts_df
            if (questions_df_override is not None and not questions_df.empty and
                    'context_ids' in questions_df.columns):
                valid_ids = questions_df['context_ids'].dropna()
                all_ids = set()
                if not valid_ids.empty:
                    is_list = valid_ids.apply(lambda x: isinstance(x, list))
                    if is_list.any():
                        all_ids = set(valid_ids[is_list].explode().unique())
                if all_ids and not self.contexts_df.empty:
                    contexts_df = self.contexts_df[self.contexts_df['context_id'].isin(all_ids)]
                else:
                    contexts_df = pd.DataFrame(columns=self.contexts_df.columns)
            result["contexts"] = self._serialize_df(contexts_df, sort_contexts_by_key)
        if include_questions:
            result["questions"] = self._serialize_df(questions_df, sort_questions_by_key)
        return result

    def _get_questions_batches(self, batch_size: int, query_string: Optional[str], with_contexts: bool,
                               sort_questions_by_key: Optional[str], sort_contexts_by_key: Optional[str]) -> List[Dict]:
        df = self.questions_df
        if query_string:
            try:
                df = df.query(query_string)
            except Exception as e:
                logger.error(f"Błąd w query_string: {query_string}", exc_info=True)
                df = pd.DataFrame(columns=self.questions_df.columns)
        if sort_questions_by_key and not df.empty:
            df = df.sort_values(by=sort_questions_by_key, kind='mergesort', na_position='last')
        num_batches = (len(df) - 1) // batch_size + 1 if not df.empty else 0
        batches = []
        for i in range(num_batches):
            batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
            batch_dict = self._get_data_as_dict(
                questions_df_override=batch_df, include_contexts=with_contexts, include_questions=True,
                sort_contexts_by_key=sort_contexts_by_key, sort_questions_by_key=sort_questions_by_key)
            batches.append(batch_dict)
        return batches

    def add_data_from_json_file(self, filepath: str,
                                dup_strategy: DuplicatesStrategy = DuplicatesStrategy.REPLACE_WITH_NEW):
        logger.info(f"Ładowanie danych z pliku JSON: {filepath} (duplikaty: {dup_strategy.value})")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Błąd podczas ładowania danych z {filepath}: {e}", exc_info=True)
            return
        self._update_dataframe('contexts_df', data.get('contexts', []), 'context_id', dup_strategy)
        self._update_dataframe('questions_df', data.get('questions', []), 'question_id', dup_strategy)
        logger.info(f"Dane z {filepath} przetworzone. Konteksty: {len(self.contexts_df)}, Pytania: {len(self.questions_df)}")

    def add_data_from_json_string(self, json_string: str,
                                  dup_strategy: DuplicatesStrategy = DuplicatesStrategy.REPLACE_WITH_NEW):
        logger.info(f"Ładowanie danych z stringa JSON (duplikaty: {dup_strategy.value})")
        try:
            data = json.loads(json_string)
        except Exception as e:
            logger.error(f"Błąd podczas ładowania danych z stringa JSON: {e}", exc_info=True)
            return
        self._update_dataframe('contexts_df', data.get('contexts', []), 'context_id', dup_strategy)
        self._update_dataframe('questions_df', data.get('questions', []), 'question_id', dup_strategy)
        logger.info(
            f"Dane ze stringa JSON przetworzone. Konteksty: {len(self.contexts_df)}, Pytania: {len(self.questions_df)}")

    def remove_questions_by_query(self, query: str):
        df = self.questions_df
        if df.empty:
            logger.warning("Nie można usunąć pytań: DataFrame jest pusty.")
            return
        try:
            to_remove = df.query(query)
            if to_remove.empty:
                logger.info(f"Brak pytań do usunięcia dla zapytania: '{query}'.")
                return
            self.questions_df.drop(index=to_remove.index, inplace=True)
            logger.info(f"Usunięto {len(to_remove)} pytań. Dataset ma teraz ma {len(self.questions_df)} pytań.")
        except Exception as e:
            logger.error("Błąd podczas usuwania pytań.", exc_info=True)

    def remove_contexts_by_query(self, query: str, by_force: bool = False):
        df = self.contexts_df
        if df.empty:
            logger.warning("Nie można usunąć kontekstów: DataFrame jest pusty.")
            return
        try:
            to_remove = df.query(query)
            if to_remove.empty:
                logger.info(f"Brak kontekstów do usunięcia dla zapytania: '{query}'.")
                return
            used_ids = set(self.questions_df[
                               'context_ids'].explode()) if not self.questions_df.empty and 'context_ids' in self.questions_df.columns else set()
            blocked = to_remove['context_id'].isin(used_ids)
            n_blocked = blocked.sum()
            if by_force:
                if n_blocked > 0:
                    logger.warning(f"Usuwane {n_blocked} kontekstów powiązanych z pytaniami (by_force=True).")
            else:
                if n_blocked > 0:
                    logger.info(
                        f"Pozostawiono {n_blocked} kontekstów powiązanych z pytaniami. Usunięto {len(to_remove) - n_blocked} kontekstów.")
                to_remove = to_remove[~blocked]
            self.contexts_df.drop(index=to_remove.index, inplace=True)
            logger.info(f"Usunięto {len(to_remove)} kontekstów. Dataset ma teraz {len(self.contexts_df)} kontekstów.")
        except Exception as e:
            logger.error("Błąd podczas usuwania kontekstów.", exc_info=True)

    def save_all_data_to_json_file(self, filepath: str, sort_contexts_by_key: Optional[str] = "context_id",
                                   sort_questions_by_key: Optional[str] = "question_id"):
        if self.contexts_df.empty and self.questions_df.empty:
            logger.info("Brak danych do zapisania. Plik nie zostanie utworzony.")
            return
        data = self._get_data_as_dict(
            sort_contexts_by_key=sort_contexts_by_key, sort_questions_by_key=sort_questions_by_key)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Dane zapisane do {filepath}")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania danych do {filepath}", exc_info=True)

    def get_all_data_as_json_string(self, sort_contexts_by_key: Optional[str] = None,
                                    sort_questions_by_key: Optional[str] = None) -> str:
        data = self._get_data_as_dict(
            sort_contexts_by_key=sort_contexts_by_key, sort_questions_by_key=sort_questions_by_key)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def get_questions_as_json_string(self, query_string: Optional[str] = None, with_contexts: bool = False,
                                     sort_questions_by_key: Optional[str] = None,
                                     sort_contexts_by_key: Optional[str] = None) -> str:
        if self.questions_df.empty:
            logger.warning("Nie można pobrać pytań: DataFrame pytań jest pusty.")
            return json.dumps(self._get_data_as_dict(
                questions_df_override=pd.DataFrame(columns=self.questions_df.columns),
                include_contexts=with_contexts, include_questions=True,
                sort_contexts_by_key=sort_contexts_by_key, sort_questions_by_key=sort_questions_by_key),
                ensure_ascii=False, indent=2)
        try:
            filtered = self.questions_df.query(query_string) if query_string else self.questions_df
            data = self._get_data_as_dict(
                questions_df_override=filtered, include_contexts=with_contexts, include_questions=True,
                sort_contexts_by_key=sort_contexts_by_key, sort_questions_by_key=sort_questions_by_key)
            logger.info(f"Pobrano {len(data.get('questions', []))} pytań dla zapytania: '{query_string}'. "
                       f"Dołączono {len(data.get('contexts', []))} kontekstów")
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Błąd podczas pobierania pytań za pomocą zapytania '{query_string}'.", exc_info=True)
            payload = {"questions": [], "error": str(e)}
            if with_contexts:
                payload["contexts"] = []
            return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_contexts_as_json_string(self, query_string: Optional[str] = None,
                                    sort_contexts_by_key: Optional[str] = None) -> str:
        if self.contexts_df.empty:
            logger.warning("Nie można pobrać kontekstów: DataFrame kontekstów jest pusty.")
            return json.dumps({"contexts": []}, ensure_ascii=False, indent=2)
        try:
            filtered = self.contexts_df.query(query_string) if query_string else self.contexts_df
            data = self._serialize_df(filtered, sort_contexts_by_key)
            logger.info(f"Pobrano {len(data)} kontekstów dla zapytania: '{query_string}'.")
            return json.dumps({"contexts": data}, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Błąd podczas pobierania kontekstów za pomocą zapytania '{query_string}'.", exc_info=True)
            return json.dumps({"contexts": [], "error": str(e)}, ensure_ascii=False, indent=2)

    def save_questions_in_batches_as_jsonl_file(self, output_filepath: str, batch_size: int = 10,
                                                query_string: Optional[str] = None, with_contexts: bool = False,
                                                sort_questions_by_key: Optional[str] = None,
                                                sort_contexts_by_key: Optional[str] = None):
        batches = self._get_questions_batches(batch_size, query_string, with_contexts,
                                             sort_questions_by_key, sort_contexts_by_key)
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for batch in batches:
                    f.write(json.dumps(batch, ensure_ascii=False) + '\n')
            logger.info(f"Pomyślnie zapisano batch'e do {output_filepath}")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania batchy do {output_filepath}", exc_info=True)

    def get_questions_in_batches_as_jsonl_string(self, batch_size: int = 10, query_string: Optional[str] = None,
                                                with_contexts: bool = False, sort_questions_by_key: Optional[str] = None,
                                                sort_contexts_by_key: Optional[str] = None) -> str:
        batches = self._get_questions_batches(batch_size, query_string, with_contexts,
                                             sort_questions_by_key, sort_contexts_by_key)
        return '\n'.join(json.dumps(batch, ensure_ascii=False) for batch in batches)

    def get_stats(self) -> dict:
        logger.debug("Zbieranie statystyk z DatasetProcessor. Pytania: %d, Konteksty: %d",
                     len(self.questions_df), len(self.contexts_df))
        return {
            "questions_count": len(self.questions_df),
            "contexts_count": len(self.contexts_df)
        }