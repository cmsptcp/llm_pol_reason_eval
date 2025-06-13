from pathlib import Path
from typing import List, Dict, Any, Union
import jinja2


class PromptManager:
    def __init__(self, templates_dir: Union[str, Path]):
        self.templates_dir = Path(templates_dir)
        print(f"Inicjalizacja PromptManager z katalogiem szablonów: {self.templates_dir}")
        if not self.templates_dir.is_dir():
            raise FileNotFoundError(f"Katalog szablonów nie istnieje: {self.templates_dir}")
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=False, trim_blocks=True, lstrip_blocks=True)

    def _get_template_path(self, model_name: str, template_name: str) -> str:
        model_specific_path = Path(model_name) / template_name
        default_path = Path('default') / template_name
        print(f"Sprawdzanie szablonu: {self.templates_dir / model_specific_path} dla modelu: {model_name}")
        if (self.templates_dir / model_specific_path).exists():
            print(f"Znaleziono szablon: {model_specific_path} dla modelu: {model_name}")
            return model_specific_path.as_posix()
        print(f"Sprawdzanie szablonu: {self.templates_dir / default_path} dla modelu: {model_name}")
        if (self.templates_dir / default_path).exists():
            print(f"Znaleziono szablon: {default_path} dla modelu: {model_name}")
            return default_path.as_posix()
        raise FileNotFoundError(f"Nie znaleziono szablonu '{template_name}' dla '{model_name}' ani w 'default'.")

    def get_question_prompt(self, model_name: str, batch_data: Dict[str, Any]) -> List[Dict[str, str]]:
        system_template_path = self._get_template_path(model_name, "system.jinja2")
        print(f"Używanie szablonu systemowego: {system_template_path} dla modelu: {model_name}")
        user_template_path = self._get_template_path(model_name, "base_question_prompt.jinja2")
        print(f"Używanie szablonu użytkownika: {user_template_path} dla modelu: {model_name}")

        # system_content = self.jinja_env.get_template(system_template_path).render()
        try:
            print(f"Próba załadowania szablonu: {system_template_path}")
            print(f"Loader szuka w katalogu: {self.jinja_env.loader.searchpath}")
            print(f"Ścieżka szablonu przekazana do get_template: {system_template_path}")
            template = self.jinja_env.get_template(system_template_path)
            print("Szablon załadowany pomyślnie.")
            system_content = template.render()
            print("Szablon wyrenderowany pomyślnie.")
        except jinja2.TemplateNotFound as e:
            print(f"Nie znaleziono szablonu: {e}")
        except Exception as e:
            print(f"Błąd podczas renderowania szablonu: {e}")

        user_content = self.jinja_env.get_template(user_template_path).render(**batch_data)

        return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]