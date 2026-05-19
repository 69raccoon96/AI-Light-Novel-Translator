import os

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SOURCE_FILE = os.path.join(INPUT_DIR, "source.txt")
SEGMENTS_FILE = os.path.join(OUTPUT_DIR, "1_segments.json")
GLOSSARY_FILE = os.path.join(OUTPUT_DIR, "2_glossary.json")
TRANSLATED_FILE = os.path.join(OUTPUT_DIR, "3_translated.json")
FINAL_FILE = os.path.join(OUTPUT_DIR, "4_final.json")

# Ollama настройки
OLLAMA_BASE_URL = "http://localhost:11434"

# Модели
MODEL_GLOSSARY  = "qwen3:14b-q4_K_M"     # извлечение терминов + строгий JSON
MODEL_TRANSLATE = "qwen3:14b-q4_K_M"     # точный перевод смысла
MODEL_PROOFREAD = "gemma3:27b"      # литературная полировка русского

# Параметры разделения
MAX_SEGMENT_CHARS = 1500
MIN_SEGMENT_CHARS = 100

# Параметры генерации
TRANSLATION_TEMPERATURE = 0.3
GLOSSARY_TEMPERATURE = 0.2
PROOFREAD_TEMPERATURE = 0.2

# Создаём директории если не существуют
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
