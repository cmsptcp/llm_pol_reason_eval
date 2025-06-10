from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class InferenceClient(ABC):
    @abstractmethod
    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        pass


class HuggingFaceClient(InferenceClient):
    def __init__(self, model_path: str, default_generation_params: Optional[Dict[str, Any]] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"HuggingFaceClient: Inicjalizacja modelu {model_path} na urządzeniu: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.default_generation_config = GenerationConfig.from_pretrained(model_path)
        if default_generation_params:
            self.default_generation_config.update(**default_generation_params)
        if self.default_generation_config.pad_token_id is None:
            self.default_generation_config.pad_token_id = self.tokenizer.eos_token_id

        print(f"HuggingFaceClient: Domyślna konfiguracja generowania: {self.default_generation_config}")

    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        config_to_use = self.default_generation_config

        if generation_params_override:
            config_to_use = self.default_generation_config.copy()
            config_to_use.update(**generation_params_override)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, generation_config=config_to_use)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response