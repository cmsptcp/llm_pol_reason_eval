import yaml
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import jinja2


class PromptManager:
    def __init__(self, templates_dir: Union[str, Path]):
        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.is_dir():
            raise FileNotFoundError(f"Katalog szablonów nie istnieje: {self.templates_dir}")

        self.jinja2_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _resolve_template_path(self, model_cfg: Dict, template_name: str) -> str:
        model_name = model_cfg['name']
        model_family = model_cfg.get('family')

        potential_paths = [
            Path(model_name) / template_name,
            Path(model_family) / template_name if model_family else None,
            Path('default') / template_name
        ]

        for path in filter(None, potential_paths):
            if (self.templates_dir / path).exists():
                print(f"Używanie szablonu: {path.as_posix()}")
                return path.as_posix()

        raise FileNotFoundError(
            f"Nie znaleziono szablonu '{template_name}' dla modelu '{model_name}', rodziny '{model_family}' ani w 'default'.")

    def get_question_prompt(self, model_cfg: Dict, batch_data: Dict, composition: Dict) -> List[Dict[str, str]]:
        main_template_name = composition.get("main_template", "base_question_prompt.jinja2")

        # Rozwiązujemy ścieżkę do szablonu systemowego i głównego, używając logiki fallback
        system_template_path = self._resolve_template_path(model_cfg, "system.jinja2")
        user_template_path = self._resolve_template_path(model_cfg, main_template_name)

        # Przygotowujemy pełny kontekst do renderowania, przekazując flagi i dane
        render_context = {
            "components": composition.get("components", []),
            "template_params": composition.get("template_params", {}),
            **batch_data
        }

        system_content = self.jinja2_env.get_template(system_template_path).render()
        user_content = self.jinja2_env.get_template(user_template_path).render(**render_context)

        return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]