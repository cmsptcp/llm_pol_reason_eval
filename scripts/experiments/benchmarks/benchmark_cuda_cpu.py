import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_names = [
    # "microsoft/Phi-4-mini-instruct", # too slow
    "microsoft/phi-1_5",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2-0.5B",
    "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-1B",
    "speakleash/Bielik-1.5B-v3.0-Instruct",
    # "speakleash/Bielik-1.5B-v3"
    ]
hf_token = os.getenv("HF_TOKEN")

def benchmark_single_model(model_name, device_str="cuda", torch_compile=False, attn_implementation=None):
    if device_str == "cuda":
        torch.cuda.empty_cache()

    # Pomiar czasu pobierania tokenizera
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    # Ustaw pad_token_id na eos_token_id dla Qwen
    if "Qwen" in model_name:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    end = time.time()
    print(f"Pobieranie tokenizera: {end - start:.2f} s")

    # Pobieranie modelu
    start = time.time()
    model_kwargs = {
        "token": hf_token,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if attn_implementation is not None and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = attn_implementation
    end = time.time()
    print(f"Pobieranie modelu: {end - start:.2f} s")
    if torch_compile:
        model = torch.compile(model)

    prompt = "To jest test wydajności. " * 10
    max_new_tokens = 256

    device = torch.device(device_str)
    model_device = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generate_kwargs = {}
    if "Qwen" in model_name:
        generate_kwargs["attention_mask"] = inputs["attention_mask"]
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    model.eval()
    with torch.no_grad():
        outputs = model_device.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample= True if model_name.startswith("Qwen") else False,
            use_cache=True,
            **generate_kwargs
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    num_generated_tokens = generated.shape[0]
    elapsed = end - start
    tokens_per_sec = num_generated_tokens / elapsed
    print(f"[{device_str.upper()}] Wygenerowano {num_generated_tokens} tokenów w {elapsed:.2f} s ({tokens_per_sec:.2f} tokenów/s)")



if __name__ == "__main__":
    # Sprawdzenie dostępności GPU
    if torch.cuda.is_available():
        print("CUDA jest dostępne.")
        device_str = "cuda"
    else:
        print("CUDA nie jest dostępne. Używam CPU.")
        device_str = "cpu"

    torch.backends.cudnn.benchmark = True
    for model_name in model_names:
        # print(f"Benchmarking modelu: {model_name}")
        # benchmark_single_model(model_name, device_str)
        print(f"Benchmarking modelu: {model_name} z torch.compile")
        benchmark_single_model(model_name, device_str, torch_compile=True)
        print(f"Benchmarking modelu: {model_name} z torch.compile i sdpa")
        benchmark_single_model(model_name, device_str, torch_compile=True, attn_implementation="sdpa")