import kaggle
import os
import json
import time
import builtins # Potrzebne do łatania open()

# --- Stałe globalne ---
KAGGLE_USERNAME = "piotrorowski"
NOTEBOOK_SLUG = "test-bielika-11b-na-kaggle" # Upewnij się, że to jest poprawny slug
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebooks", "experiments", "environment_test_kaggle_hf")
METADATA_FILENAME = "kernel-metadata.json"

# --- Mechanizm łatania builtins.open ---
_true_original_builtins_open = builtins.open

def _patched_open_for_kaggle_api(*args, **kwargs):
    """
    Nakładka na builtins.open, która domyślnie używa UTF-8 dla trybu odczytu tekstowego,
    jeśli kodowanie nie zostało określone przez wywołującego.
    """
    mode = kwargs.get('mode', args[1] if len(args) > 1 and isinstance(args[1], str) else 'r')
    is_text_read_mode = 'b' not in mode and ('r' in mode or 'a' in mode or '+' in mode or 'w' in mode) # rozszerzono na inne tryby tekstowe

    if is_text_read_mode:
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
            # print(f"DEBUG: Monkey-patch: Wymuszono UTF-8 dla {args[0]} (tryb: {mode})")
    return _true_original_builtins_open(*args, **kwargs)

def apply_kaggle_encoding_patch():
    """Stosuje łatkę na builtins.open."""
    # print("DEBUG: Stosowanie łatki builtins.open dla Kaggle API.")
    builtins.open = _patched_open_for_kaggle_api

def revert_kaggle_encoding_patch():
    """Przywraca oryginalną funkcję builtins.open."""
    # print("DEBUG: Przywracanie oryginalnej funkcji builtins.open.")
    builtins.open = _true_original_builtins_open
# --- Koniec mechanizmu łatania ---

def update_kernel_id_in_metadata(metadata_path, username, slug):
    """Aktualizuje pole 'id' w pliku kernel-metadata.json."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        expected_id = f"{username}/{slug}"
        if metadata.get('id') == expected_id:
            print(f"Pole 'id' w {metadata_path} jest poprawne: {expected_id}")
            return

        metadata['id'] = expected_id
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Zaktualizowano 'id' w {metadata_path} na: {expected_id}")
    except Exception as e:
        print(f"Błąd podczas aktualizacji kernel-metadata.json: {e}")
        raise

def push_and_run_notebook(notebook_directory_path, kernel_user, kernel_slug):
    """Wysyła notatnik na Kaggle i próbuje monitorować jego uruchomienie."""
    full_kernel_id = f"{kernel_user}/{kernel_slug}"
    original_cwd = os.getcwd() # Zapamiętaj oryginalny katalog roboczy

    apply_kaggle_encoding_patch() # Zastosuj łatkę przed operacjami Kaggle API

    try:
        print(f"Próba wysłania notatnika z katalogu: {os.path.abspath(notebook_directory_path)} jako {full_kernel_id}")

        # Zmień bieżący katalog na katalog notatnika
        os.chdir(notebook_directory_path)

        print("Wysyłanie kernela na Kaggle (to może chwilę potrwać)...")
        # Użyj "." jako folder, ponieważ jesteśmy już w odpowiednim katalogu
        kaggle.api.kernels_push(folder=".")

        # Powrót do pierwotnego katalogu nie jest tu konieczny,
        # ponieważ operacje status/pull działają na ID kernela.
        # Główny powrót będzie w bloku finally.
        # Jeśli jednak jakaś operacja poniżej wymagałaby oryginalnego CWD, można by tu wrócić:
        # if os.getcwd() != original_cwd:
        #     os.chdir(original_cwd)

        print(f"Notatnik '{full_kernel_id}' został pomyślnie wysłany/zaktualizowany na Kaggle.")
        print(f"Link do kernela: https://www.kaggle.com/{kernel_user}/{kernel_slug}/code")

        # Monitorowanie statusu kernela
        print("\nRozpoczęcie monitorowania statusu kernela...")
        timeout_seconds = 600
        check_interval = 30
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                status = kaggle.api.kernels_status(kernel=full_kernel_id)
                kernel_status_str = status.get('status', 'nieznany')
                latest_version = status.get('latestKernelVersionNumber', 'N/A')
                print(f"[{time.strftime('%H:%M:%S')}] Status kernela '{full_kernel_id}' (ver: {latest_version}): {kernel_status_str}")

                if kernel_status_str in ['complete', 'error', 'cancelled']:
                    print(f"Kernel zakończył pracę ze statusem: {kernel_status_str}")
                    # Ścieżka do zapisu outputu względem oryginalnego katalogu roboczego
                    output_path_base = os.path.join(original_cwd, "kaggle_output")
                    output_path_kernel = os.path.join(output_path_base, kernel_slug)

                    if not os.path.exists(output_path_kernel):
                        os.makedirs(output_path_kernel)
                    print(f"Pobieranie wyników kernela do katalogu: {output_path_kernel}")
                    try:
                        kaggle.api.kernels_pull(kernel=full_kernel_id, path=output_path_kernel, metadata=False)
                        print(f"Pliki wynikowe (w tym notatnik z outputem) powinny być w {output_path_kernel}")
                    except Exception as e_out:
                        print(f"Nie udało się pobrać wyników/logów dla '{full_kernel_id}': {e_out}")
                    break
                elif kernel_status_str == 'running' and status.get('failureReason') == 'EXECUTION_TIMEOUT':
                    print(f"Kernel przekroczył limit czasu wykonania na Kaggle.")
                    break
            except Exception as e_status:
                print(f"Błąd podczas sprawdzania statusu kernela '{full_kernel_id}': {e_status}")
            time.sleep(check_interval)
        else:
            print(f"Osiągnięto limit czasu ({timeout_seconds}s) oczekiwania na zakończenie kernela. Sprawdź status ręcznie na Kaggle.")

    except Exception as e:
        print(f"Błąd podczas wysyłania lub uruchamiania notatnika na Kaggle: {e}")
        print("Upewnij się, że masz poprawnie skonfigurowane Kaggle API (plik kaggle.json).")
        print("Sprawdź również, czy `kernel-metadata.json` jest poprawnie sformatowany.")
        if '403 Client Error: Forbidden for url' in str(e) or 'EnsureEscrowConstraints' in str(e):
            print("Błąd 403 lub EnsureEscrowConstraints: Sprawdź, czy Twoje konto Kaggle ma zweryfikowany numer telefonu.")
            print("Jest to często wymagane do korzystania z GPU lub internetu w kernelach.")
    finally:
        revert_kaggle_encoding_patch() # Zawsze przywracaj oryginalną funkcję open
        # Upewnij się, że katalog roboczy jest przywrócony
        if os.getcwd() != original_cwd:
            os.chdir(original_cwd)
        # print(f"DEBUG: Przywrócono CWD do: {os.getcwd()}")


if __name__ == "__main__":
    # Uwierzytelnienie Kaggle API
    try:
        kaggle.api.authenticate()
        print("Uwierzytelnienie Kaggle API powiodło się.")
    except Exception as e:
        print(f"Błąd uwierzytelniania Kaggle API: {e}")
        print("Upewnij się, że plik kaggle.json znajduje się w ~/.kaggle/ (Linux/macOS) lub C:\\Users\\<USER>\\.kaggle\\ (Windows).")
        exit(1)

    print(f"Wypisz bieżący katalog: {os.getcwd()}")

    # Pełna ścieżka do pliku metadata.json
    metadata_file_path = os.path.join(NOTEBOOK_DIR, METADATA_FILENAME)
    print(f"Sprawdzam ścieżkę: {metadata_file_path}")
    if not os.path.isfile(metadata_file_path):
        print(f"BŁĄD: Plik {metadata_file_path} nie istnieje!")
        exit(1)
    print("Plik istnieje: True")


    # Krok 1: Zaktualizuj ID w kernel-metadata.json (na wszelki wypadek)
    try:
        update_kernel_id_in_metadata(metadata_file_path, KAGGLE_USERNAME, NOTEBOOK_SLUG)
    except Exception:
        exit(1) # Zakończ, jeśli aktualizacja metadanych się nie powiedzie

    # Krok 2: Wyślij notatnik na Kaggle i monitoruj
    push_and_run_notebook(NOTEBOOK_DIR, KAGGLE_USERNAME, NOTEBOOK_SLUG)

    print("\nSkrypt zakończył działanie.")