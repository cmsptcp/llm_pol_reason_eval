import yaml
import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root / "src"))

from llm_pol_reason_eval.qa_engine.llm_qa_engine import LLMQAEngine
from llm_pol_reason_eval.qa_engine.inference_client import HuggingFaceClient

def main(run_config_path: str, experiment_name: str):
    run_config_path = project_root / run_config_path
    with open(run_config_path, 'r', encoding='utf-8') as f:
        run_config = yaml.safe_load(f)['experiments'].get(experiment_name)
    if not run_config:
        raise ValueError(f"Nie znaleziono eksperymentu '{experiment_name}' w pliku {run_config_path}")

    models_config_path = project_root / "config/models.yaml"
    with open(models_config_path, 'r', encoding='utf-8') as f:
        models_config = yaml.safe_load(f)

    print(f"--- Uruchamianie eksperymentu: {experiment_name} ---")

    input_dataset_path = project_root / run_config['input_dataset']
    output_dir = project_root / run_config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    model_key = run_config['model']
    model_cfg = models_config.get(model_key)

    print(f"\n--- Przetwarzanie modelem: {model_key} ---")

    run_overrides = run_config.get("param_overrides", {})
    final_gen_params = model_cfg['generation_params'].copy()
    final_gen_params.update(run_overrides.get('default', {}))

    inference_client = HuggingFaceClient(
        model_path=model_cfg['path'],
        default_generation_params=final_gen_params
    )
    engine = LLMQAEngine(model_name=model_key, model_path=model_cfg['path'], inference_client=inference_client)

    engine.generate_answers(
        dataset_filepath=str(input_dataset_path),
        param_overrides=run_overrides
    )

    output_path = output_dir / f"answers_{experiment_name}.json"
    engine.save_results(output_filepath=str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generuj odpowiedzi na pytania używając LLM.")
    parser.add_argument("run_config", type=str, help="Ścieżka do pliku YAML z konfiguracją uruchomień.")
    parser.add_argument("experiment_name", type=str, help="Nazwa eksperymentu do uruchomienia.")
    args = parser.parse_args()
    main(args.run_config, args.experiment_name)