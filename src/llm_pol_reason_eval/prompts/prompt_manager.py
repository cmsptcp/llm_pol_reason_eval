from pathlib import Path
from typing import List, Dict, Any, Union
import jinja2


class PromptManager:
    def __init__(self, templates_dir: Union[str, Path]):
        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.is_dir():
            raise FileNotFoundError(f"Katalog szablonów nie istnieje: {self.templates_dir}")
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=False, trim_blocks=True, lstrip_blocks=True)

    def _get_template_path(self, model_name: str, template_name: str) -> str:
        model_specific_path = Path(model_name) / template_name
        default_path = Path('default') / template_name
        if (self.templates_dir / model_specific_path).exists():
            return str(model_specific_path)
        if (self.templates_dir / default_path).exists():
            return str(default_path)
        raise FileNotFoundError(f"Nie znaleziono szablonu '{template_name}' dla '{model_name}' ani w 'default'.")

    def get_question_prompt(self, model_name: str, batch_data: Dict[str, Any]) -> List[Dict[str, str]]:
        system_template_path = self._get_template_path(model_name, "system.jinja2")
        user_template_path = self._get_template_path(model_name, "base_question_prompt.jinja2")

        system_content = self.env.get_template(system_template_path).render()
        user_content = self.env.get_template(user_template_path).render(**batch_data)

        return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]