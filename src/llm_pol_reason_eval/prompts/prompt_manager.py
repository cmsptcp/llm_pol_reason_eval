import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import jinja2


class PromptManager:

    def __init__(self, templates_dir: Union[str, Path]):

        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.is_dir():
            raise FileNotFoundError(f"Katalog szablonów nie istnieje: {self.templates_dir}")

        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=False,  # Ważne dla generowania tekstu, a nie HTML
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _get_template_path(self, model_name: str, question_type: str, question_category: str,
                           template_name: str) -> Path:
        # Prosta logika fallback: najpierw szukaj w folderze specyficznym dla modelu, potem w 'default'
        model_specific_path = Path(model_name) / template_name
        default_path = Path('default') / template_name

        potential_paths = [model_specific_path, default_path]

        for path in potential_paths:
            if (self.templates_dir / path).exists():
                return path

        raise FileNotFoundError(f"Nie znaleziono szablonu '{template_name}' dla modelu '{model_name}' ani w 'default'.")

    def _render_prompt_content(self, template_path: Path, **kwargs) -> str:
        template = self.env.get_template(str(template_path))
        return template.render(**kwargs)

    def get_single_question_prompt(
            self,
            model_name: str,
            question_type: str,
            question_category: str,
            question_text: str,
            context_texts: Optional[List[str]] = None,
            choices: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:

        # 1. Wybierz szablon dla treści użytkownika
        user_template_path = self._get_template_path(
            model_name, question_type, question_category, "base_single.jinja2"
        )

        # 2. Renderuj zawartość
        system_content = self._render_prompt_content(Path("system.jinja2"))
        user_content = self._render_prompt_content(
            user_template_path,
            question_text=question_text,
            context_texts=context_texts or [],
            choices=choices or []
        )

        # 3. Zwróć strukturę czatu
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def get_multiple_question_prompt(
            self,
            model_name: str,
            question_type: str,
            question_category: str,
            question_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:

        # 1. Wybierz szablon
        user_template_path = self._get_template_path(
            model_name, question_type, question_category, "base_multiple.jinja2"
        )

        # 2. Renderuj zawartość
        system_content = self._render_prompt_content(Path("system.jinja2"))
        # Przekazujemy całą listę danych do szablonu, który ma pętlę
        user_content = self._render_prompt_content(
            user_template_path,
            questions=question_data_list
        )

        # 3. Zwróć strukturę czatu
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]