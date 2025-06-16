from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline


class InferenceClient(ABC):
    @abstractmethod
    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        """Przetwarza pojedynczy prompt i zwraca pojedynczą odpowiedź."""
        pass

    @abstractmethod
    def get_responses_with_batching(self, prompts: List[str],
                                    generation_params_override: Optional[Dict[str, Any]] = None) -> List[str]:
        """Przetwarza listę promptów naraz (batching) i zwraca listę odpowiedzi."""
        pass


class HuggingFaceClient(InferenceClient):
    def __init__(self, model_path: str, default_generation_params: Optional[Dict[str, Any]] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"HuggingFaceClient: Inicjalizacja modelu {model_path} na urządzeniu: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device, trust_remote_code=True
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.default_generation_config = GenerationConfig.from_pretrained(model_path)
        if default_generation_params:
            self.default_generation_config.update(**default_generation_params)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.default_generation_config.pad_token_id = self.tokenizer.eos_token_id

        print(f"HuggingFaceClient: Domyślna konfiguracja generowania: {self.default_generation_config}")

    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        return self.get_responses_with_batching([prompt], generation_params_override)[0]

    def get_responses_with_batching(self, prompts: List[str],
                                    generation_params_override: Optional[Dict[str, Any]] = None) -> List[str]:
        config_to_use = self.default_generation_config
        if generation_params_override:
            config_to_use = self.default_generation_config.copy()
            config_to_use.update(**generation_params_override)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=config_to_use)

        input_length = inputs.input_ids.shape[1]
        return [self.tokenizer.decode(output[input_length:], skip_special_tokens=True) for output in outputs]


class HuggingFacePipelineClient(InferenceClient):
    def __init__(self, model_path: str, default_generation_params: Optional[Dict[str, Any]] = None,
                 task: str = "text-generation"):
        self.device_id = 0 if torch.cuda.is_available() else -1
        print(
            f"HuggingFacePipelineClient: Inicjalizacja pipeline dla {model_path} na {'cuda:0' if self.device_id == 0 else 'cpu'}")

        self.text_generator = pipeline(
            task, model=model_path, tokenizer=model_path, device=self.device_id, trust_remote_code=True
        )
        self.default_generation_params = default_generation_params if default_generation_params else {}

        if self.text_generator.tokenizer.pad_token_id is None:
            self.text_generator.tokenizer.pad_token_id = self.text_generator.tokenizer.eos_token_id

        print(f"HuggingFacePipelineClient: Domyślne parametry generowania: {self.default_generation_params}")

    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        return self.get_responses_with_batching([prompt], generation_params_override)[0]

    def get_responses_with_batching(self, prompts: List[str],
                                    generation_params_override: Optional[Dict[str, Any]] = None) -> List[str]:
        params_to_use = self.default_generation_params.copy()
        if generation_params_override:
            params_to_use.update(generation_params_override)

        generated_outputs = self.text_generator(prompts, **params_to_use)

        responses = []
        for i, output_group in enumerate(generated_outputs):
            if output_group and isinstance(output_group, list) and 'generated_text' in output_group[0]:
                response_text = output_group[0]['generated_text']
                prompt = prompts[i]
                responses.append(
                    response_text[len(prompt):].strip() if response_text.startswith(prompt) else response_text.strip()
                )
            else:
                print(f"Ostrzeżenie: Otrzymano nieoczekiwany format odpowiedzi z pipeline: {output_group}")
                responses.append("")

        return responses