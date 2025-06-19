from typing import Dict, Any
import logging
from .inference_client import InferenceClient, HuggingFaceClient, HuggingFacePipelineClient
try:
    from .inference_client import VLLMClient
except ImportError:
    VLLMClient = None

def create_inference_client(model_cfg: Dict[str, Any]) -> InferenceClient:
    """
    Fabryka tworząca odpowiedniego klienta inferencji na podstawie konfiguracji modelu.
    Args:
        model_cfg: Słownik z konfiguracją modelu, wczytany z pliku models.yaml.
    Returns:
        Instancja odpowiedniego klienta dziedziczącego po InferenceClient.
    """
    model_path = model_cfg['path']
    default_gen_params = model_cfg.get('generation_params', {})

    backend = model_cfg.get('inference_backend', 'huggingface')
    print(f"Tworzenie klienta dla modelu '{model_cfg['name']}' z backendem: {backend.upper()}")

    if backend == 'vllm':
        if VLLMClient is None:
            raise ImportError(
                "Backend 'vllm' jest wymagany, ale biblioteka vLLM nie jest zainstalowana. Uruchom 'pip install vllm'.")
        return VLLMClient(
            model_path=model_path,
            model_config=model_cfg,
            default_generation_params=default_gen_params
        )

    elif backend == 'huggingface':
        return HuggingFaceClient(
            model_path=model_path,
            model_config=model_cfg,
            default_generation_params=default_gen_params
        )

    elif backend == 'huggingface_pipeline':
        return HuggingFacePipelineClient(
            model_path=model_path,
            model_config=model_cfg,
            default_generation_params=default_gen_params
        )
    else:
        raise ValueError(f"Nieznany backend inferencji w konfiguracji modelu: '{backend}'")