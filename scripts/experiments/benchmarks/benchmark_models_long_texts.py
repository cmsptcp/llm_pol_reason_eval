import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.llm_pol_reason_eval.test_data.long_texts import LONG_TEXT_LUECKE

model_names = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "speakleash/Bielik-1.5B-v3.0-Instruct",
    ]
hf_token = os.getenv("HF_TOKEN")

def benchmark_text_synthesis_single_model(model_name, input_text=LONG_TEXT_LUECKE, device_str="cuda", torch_compile=False):
    if device_str == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    end = time.time()
    print(f"Pobieranie modelu: {end - start:.2f} s")
    if torch_compile:
        model = torch.compile(model)

    max_new_tokens = 1024

    device = torch.device(device_str)
    model = model.to(device)
    instruction = "Proszę podać streszczenie poniższego tekstu w 100 słowach:\n\n"
    inputs = tokenizer(instruction+input_text, return_tensors="pt").to(device)

    # Wypisanie liczby tokenów wejściowych
    input_length = inputs["input_ids"].shape[1]
    print(f"Liczba tokenów wejściowych: {input_length}")

    # generate_kwargs = {}
    # if "Qwen" in model_name:
    #    generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    model.eval()
    # Pomiar czasu generacji
    if device_str == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if "Qwen" in model_name else False,
            temperature=0.7,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    if device_str == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    generated = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    num_generated_tokens = generated.shape[0]
    elapsed = end - start
    tokens_per_sec = num_generated_tokens / elapsed
    print(f"[{device_str.upper()}] Wygenerowano {num_generated_tokens} tokenów w {elapsed:.2f} s ({tokens_per_sec:.2f} tokenów/s)")
    print(f"Wygenerowany tekst: {generated_text}")

    return generated_text


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
        try:
            print(f"Benchmarking modelu z długim tekstem: {model_name} z torch.compile")
            benchmark_text_synthesis_single_model(model_name, LONG_TEXT_LUECKE, device_str, torch_compile=True)
        except Exception as e:
            print(f"Błąd dla modelu {model_name}: {e}")
            continue


