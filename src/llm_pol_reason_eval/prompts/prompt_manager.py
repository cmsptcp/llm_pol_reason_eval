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
                return path.as_posix()

        raise FileNotFoundError(
            f"Nie znaleziono szablonu '{template_name}' dla modelu '{model_name}', rodziny '{model_family}' ani w 'default'.")

    def prepare_question_chatml_prompt(
            self,
            model_cfg: Dict,
            question_data: Dict,
            contexts_for_question: Dict[str, Dict],
            composition: Dict
    ) -> List[Dict[str, str]]:
        main_template_name = composition.get("main_template", "base_question_prompt.jinja2")
        system_template_path = self._resolve_template_path(model_cfg, "system.jinja2")
        user_template_path = self._resolve_template_path(model_cfg, main_template_name)

        template_params = composition.get("template_params", {}).copy()

        if "few_shot" in composition.get("components", []):
            question_type = question_data.get("question_type")
            resolved_few_shot_path = None

            if question_type:
                specific_path = Path("_components") / question_type / "examples_few_shot.jinja2"
                if (self.templates_dir / specific_path).exists():
                    resolved_few_shot_path = specific_path.as_posix()

            if not resolved_few_shot_path:
                default_path = Path("_components") / "default" / "examples_few_shot.jinja2"
                if (self.templates_dir / default_path).exists():
                    resolved_few_shot_path = default_path.as_posix()

            if resolved_few_shot_path:
                template_params["few_shot_path"] = resolved_few_shot_path

        render_context = {
            "components": composition.get("components", []),
            "template_params": template_params,
            "question": question_data,
            "contexts": contexts_for_question
        }

        system_content = self.jinja2_env.get_template(system_template_path).render()
        user_content = self.jinja2_env.get_template(user_template_path).render(**render_context)

        return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    def prepare_question_chatml_prompt_batch(
            self,
            model_cfg: Dict,
            batch_data: Dict,
            composition: Dict
    ) -> List[List[Dict[str, str]]]:
        all_prompts_chatml = []
        questions_dict = batch_data.get('questions', {})
        all_contexts_dict = batch_data.get('contexts', {})

        base_template_params = model_cfg.get('prompt_template_params', {})
        run_template_params = composition.get('template_params', {})

        final_template_params = {**base_template_params, **run_template_params}

        final_composition = composition.copy()
        final_composition['template_params'] = final_template_params

        for i, (q_id, q_data) in enumerate(questions_dict.items()):
            contexts_for_this_question = {
                cid: all_contexts_dict[cid]
                for cid in q_data.get("context_ids", [])
                if cid in all_contexts_dict
            }

            chatml_prompt = self.prepare_question_chatml_prompt(
                model_cfg,
                q_data,
                contexts_for_this_question,
                final_composition
            )
            all_prompts_chatml.append(chatml_prompt)
        return all_prompts_chatml