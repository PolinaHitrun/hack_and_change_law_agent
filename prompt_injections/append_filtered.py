import os
import re
import pandas as pd

# --- КОНФИГУРАЦИЯ ---
TXT_FILE = "injections.txt"
DATASET_FILE = "combined_dataset_translated_ru.csv"  # ваш переведенный CSV (или .jsonl)

def normalize_line(line: str) -> str:
    """Очищает строку от лишних концевых пробелов и экранированных переносов."""
    if not isinstance(line, str):
        return ""
    # Заменяем реальные переносы строк внутри одного промпта на пробел или \n (для сохранения формата 1 строка = 1 пример)
    line = line.replace("\r\n", " ").replace("\n", " ")
    return line.strip()

def append_filtered_texts():
    if not os.path.exists(DATASET_FILE):
        print(f"Ошибка: файл датасета '{DATASET_FILE}' не найден.")
        return

    # 1. Загрузка существующего TXT файла для проверки дубликатов
    existing_lines = set()
    if os.path.exists(TXT_FILE):
        with open(TXT_FILE, "r", encoding="utf-8") as f:
            for l in f:
                cleaned = l.strip()
                if cleaned:
                    existing_lines.add(cleaned)
        print(f"В '{TXT_FILE}' уже есть {len(existing_lines)} записей.")
    else:
        print(f"Файл '{TXT_FILE}' не найден, будет создан новый.")

    # 2. Загрузка переведенного датасета
    if DATASET_FILE.endswith(".csv"):
        df = pd.read_csv(DATASET_FILE)
    elif DATASET_FILE.endswith(".jsonl"):
        df = pd.read_json(DATASET_FILE, lines=True)
    else:
        df = pd.read_csv(DATASET_FILE)

    # Автоопределение нужных колонок
    en_col = next((c for c in ["text", "prompt", "en", "original_text"] if c in df.columns), None)
    ru_col = next((c for c in ["text_ru", "prompt_ru", "ru", "translated_text"] if c in df.columns), None)

    if not en_col:
        print(f"Ошибка: не найдена колонка с английским текстом. Доступные: {list(df.columns)}")
        return

    print(f"Используем колонку английского текста: '{en_col}'")
    if ru_col:
        print(f"Используем колонку русского перевода: '{ru_col}'")
    else:
        print("Внимание: колонка с русским переводом не найдена, будут обработаны только английские тексты.")

    new_entries = []
    skipped_cant_translate = 0
    skipped_duplicates = 0

    # Регулярка для фильтрации "не могу" (в любом регистре)
    cant_regex = re.compile(r"не\s+могу", re.IGNORECASE)

    for _, row in df.iterrows():
        # --- Обработка английского текста ---
        en_text = normalize_line(row.get(en_col, ""))
        if en_text:
            if en_text not in existing_lines:
                new_entries.append(en_text)
                existing_lines.add(en_text)
            else:
                skipped_duplicates += 1

        # --- Обработка русского текста ---
        if ru_col and pd.notna(row.get(ru_col)):
            ru_text = normalize_line(row.get(ru_col, ""))
            if ru_text:
                # Проверяем на наличие "не могу"
                if cant_regex.search(ru_text):
                    skipped_cant_translate += 1
                    continue

                if ru_text not in existing_lines:
                    new_entries.append(ru_text)
                    existing_lines.add(ru_text)
                else:
                    skipped_duplicates += 1

    # 3. Дозапись в конец файла
    if new_entries:
        # Проверяем, заканчивается ли существующий файл на перевод строки
        needs_leading_newline = False
        if os.path.exists(TXT_FILE) and os.path.getsize(TXT_FILE) > 0:
            with open(TXT_FILE, "rb") as f:
                f.seek(-1, os.SEEK_END)
                last_char = f.read(1)
                if last_char not in (b"\n", b"\r"):
                    needs_leading_newline = True

        with open(TXT_FILE, "a", encoding="utf-8") as f:
            if needs_leading_newline:
                f.write("\n")
            for item in new_entries:
                f.write(item + "\n")

        print("\n--- Итоги добавления ---")
        print(f"Успешно дозаписано новых строк: {len(new_entries)}")
        print(f"Отсеяно русских отказов (содержат 'не могу'): {skipped_cant_translate}")
        print(f"Пропущено дубликатов: {skipped_duplicates}")
        print(f"Итого строк в '{TXT_FILE}': {len(existing_lines)}")
    else:
        print("\nНовых подходящих записей для добавления не найдено.")

if __name__ == "__main__":
    append_filtered_texts()