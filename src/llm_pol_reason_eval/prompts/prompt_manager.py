from pathlib import Path
from typing import List, Dict, Any, Union
import jinja2


class PromptManager:
    def __init__(self, templates_dir: Union[str, Path]):
        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.is_dir():
            raise FileNotFoundError(f"Katalog szablonów nie istnieje: {self.templates_dir}")

        self.jinja2_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir.as_posix()),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _render_turn(self, template_name: str, context: Dict) -> str:
        template = self.jinja2_env.get_template(template_name)
        return template.render(**context)

    def _resolve_template_path(self, model_cfg: Dict, template_name: str) -> str:
        model_family = model_cfg.get('family')
        potential_paths = [Path(model_family) / template_name if model_family else None,
                           Path('default') / template_name]
        for path in filter(None, potential_paths):
            if (self.templates_dir / path).exists():
                return path.as_posix()
        raise FileNotFoundError(f"Nie znaleziono szablonu '{template_name}'.")

    def prepare_conversation_prompt(self, model_cfg: Dict, question_data: Dict, contexts: Dict,
                                    composition: Dict, few_shot_examples: List[Dict[str, Any]]
                                    ) -> List[Dict[str, str]]:
        chat_history = []
        template_params = composition.get('template_params', {}).copy()
        template_params['components'] = composition.get('components', [])

        system_template_path = self._resolve_template_path(model_cfg, "system.jinja2")
        system_content = self.jinja2_env.get_template(system_template_path).render(template_params=template_params)
        chat_history.append({"role": "system", "content": system_content})

        for i, example in enumerate(few_shot_examples):
            params_for_example = {**template_params, 'question_index': i + 1}
            user_turn = self._render_turn(
                "default/user_turn.jinja2",
                {"question": example['question'], "contexts": example['contexts'],
                 "template_params": params_for_example}
            )
            assistant_turn = self._render_turn(
                "default/assistant_turn.jinja2",
                {"question": example['question'], "template_params": params_for_example}
            )
            chat_history.extend(
                [{"role": "user", "content": user_turn}, {"role": "assistant", "content": assistant_turn}])

        final_question_params = {
            **template_params,
            'question_index': len(few_shot_examples) + 1,
            'final_question': True
        }
        final_user_turn = self._render_turn(
            "default/user_turn.jinja2",
            {"question": question_data, "contexts": contexts, "template_params": final_question_params}
        )
        chat_history.append({"role": "user", "content": final_user_turn})

        return chat_history