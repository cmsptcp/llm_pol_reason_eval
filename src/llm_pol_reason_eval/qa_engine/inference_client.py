from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig
import bitsandbytes
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


class InferenceClient(ABC):
    @abstractmethod
    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        """Pobiera odpowiedź na pojedynczy prompt."""
        pass

    @abstractmethod
    def get_responses_with_batching(self, prompts: List[str],
                                    generation_params_override: Optional[Dict[str, Any]] = None) -> List[str]:
        """Pobiera odpowiedzi na listę promptów z batchingiem."""
        pass


class HuggingFaceClient(InferenceClient):
    def __init__(self, model_path: str, model_config: Optional[Dict[str, Any]] = None, default_generation_params: Optional[Dict[str, Any]] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"HuggingFaceClient: Inicjalizacja modelu {model_path} na urządzeniu: {self.device}")

        model_load_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "sdpa"
        }

        quantize_int8 = model_config and model_config.get("quantization") == "int8"

        if quantize_int8:
            print(f"HuggingFaceClient: Włączanie kwantyzacji INT8 dla {model_path}")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model_load_kwargs["quantization_config"] = quantization_config
            model_load_kwargs["device_map"] = "auto"
        else:
            model_load_kwargs["device_map"] = self.device
            model_load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_load_kwargs
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.default_generation_config = GenerationConfig.from_pretrained(model_path)
        if default_generation_params:
            self.default_generation_config.update(**default_generation_params)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if hasattr(self.default_generation_config, 'pad_token_id'):
                 self.default_generation_config.pad_token_id = self.tokenizer.eos_token_id

        print(f"HuggingFaceClient: Domyślna konfiguracja generowania: {self.default_generation_config}")

    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        return self.get_responses_with_batching([prompt], generation_params_override)[0]

    def get_responses_with_batching(self, prompts: List[str],
                                    generation_params_override: Optional[Dict[str, Any]] = None) -> List[str]:
        config_dict = self.default_generation_config.to_dict()
        if generation_params_override:
            config_dict.update(generation_params_override)
        config_to_use = GenerationConfig.from_dict(config_dict)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device if hasattr(self.model, "device") else self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=config_to_use)

        input_length = inputs.input_ids.shape[1]
        return [self.tokenizer.decode(output[input_length:], skip_special_tokens=True) for output in outputs]


class HuggingFacePipelineClient(InferenceClient):
    def __init__(self, model_path: str, model_config: Optional[Dict[str, Any]] = None, default_generation_params: Optional[Dict[str, Any]] = None,
                 task: str = "text-generation"):

        pipeline_device_id = 0 if torch.cuda.is_available() else -1
        print(f"HuggingFacePipelineClient: Inicjalizacja pipeline dla {model_path}")

        pipeline_model_kwargs = {}
        quantize_int8 = model_config and model_config.get("quantization") == "int8"
        effective_device = None

        if quantize_int8:
            print(f"HuggingFacePipelineClient: Włączanie kwantyzacji INT8 dla {model_path} w pipeline.")
            pipeline_model_kwargs["load_in_8bit"] = True
            pipeline_model_kwargs["device_map"] = "auto"
            # Gdy device_map jest używane, device w pipeline() powinno być None
        else:
            effective_device = pipeline_device_id

        if effective_device is not None:
             print(f"HuggingFacePipelineClient: Używane urządzenie: {'cuda:0' if effective_device == 0 else 'cpu'}")

        self.text_generator = pipeline(
            task,
            model=model_path,
            tokenizer=model_path,
            device=effective_device,
            trust_remote_code=True,
            model_kwargs=pipeline_model_kwargs if pipeline_model_kwargs else None
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

        if "pad_token_id" not in params_to_use and self.text_generator.tokenizer.pad_token_id is not None:
            params_to_use["pad_token_id"] = self.text_generator.tokenizer.pad_token_id

        generated_outputs = self.text_generator(prompts, **params_to_use)

        responses = []
        for i, output_group in enumerate(generated_outputs):
            if output_group and isinstance(output_group, list) and 'generated_text' in output_group[0]:
                response_text = output_group[0]['generated_text']
                current_prompt = prompts[i]
                if response_text.startswith(current_prompt):
                    responses.append(response_text[len(current_prompt):].strip())
                else:
                    responses.append(response_text.strip())
            else:
                print(f"Ostrzeżenie: Otrzymano nieoczekiwany format odpowiedzi z pipeline: {output_group}")
                responses.append("")
        return responses

class VLLMClient(InferenceClient):
    def __init__(self, model_path: str, model_config: Optional[Dict[str, Any]] = None, default_generation_params: Optional[Dict[str, Any]] = None):
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM nie jest zainstalowany. Uruchom 'pip install vllm' aby go zainstalować.")

        print(f"VLLMClient: Inicjalizacja modelu {model_path}")
        self.default_generation_params = default_generation_params if default_generation_params else {}

        # vLLM wymaga `trust_remote_code=True` dla modeli takich jak Qwen
        # Ustawiamy również `gpu_memory_utilization`, aby nie zająć całej pamięci VRAM od razu.
        # GPU L4 ma 24GB VRAM, więc 0.9 to bezpieczna wartość.
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"VLLMClient: Model {model_path} załadowany.")
        print(f"VLLMClient: Domyślne parametry generowania: {self.default_generation_params}")

    def get_response(self, prompt: str, generation_params_override: Optional[Dict[str, Any]] = None) -> str:
        return self.get_responses_with_batching([prompt], generation_params_override)[0]

    def get_responses_with_batching(self, prompts: List[str],
                                    generation_params_override: Optional[Dict[str, Any]] = None) -> List[str]:
        params_to_use = self.default_generation_params.copy()
        if generation_params_override:
            params_to_use.update(generation_params_override)

        # Konwertuj parametry na obiekt vLLM SamplingParams
        # Pomiń `do_sample`, ponieważ vLLM automatycznie go obsługuje na podstawie temperatury
        sampling_params_kwargs = {k: v for k, v in params_to_use.items() if k != 'do_sample'}
        sampling_params = SamplingParams(**sampling_params_kwargs)

        print(f"VLLMClient: Generowanie dla {len(prompts)} promptów z parametrami: {sampling_params}")
        outputs = self.llm.generate(prompts, sampling_params)

        # vLLM zwraca listę obiektów RequestOutput, z których wyciągamy tekst
        responses = [output.outputs[0].text.strip() for output in outputs]
        return responses
