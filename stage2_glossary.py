import json
import re
import argparse
import sys
import requests
import time
import os
from config import *


def call_ollama(prompt: str, model: str = MODEL_GLOSSARY,
                temperature: float = GLOSSARY_TEMPERATURE,
                num_predict: int = 4096) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict}
    }
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        print("ОШИБКА: Не удаётся подключиться к Ollama. Запустите: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return ""


def extract_terms_from_batch(segments_text: str) -> str:
    prompt = f"""You are extracting a translation glossary from a Korean light novel.

YOUR TASK: Extract ONLY proper nouns and unique fictional terms that a translator must keep consistent across chapters.

EXTRACT (these are good):
- Character names (人名): 강진호, 유더 바이엘, 코델리아 체이스, 마이아
- Place names: 플레이아데스, 세일룬 왕국, 벨카인 산맥, 바이엘 백작가
- Organizations / families: 체이스 백작가, 태양신의 사제
- Titles & ranks: 백작, 변경백, 성기사
- Game / world specific terms: 영웅전기2, 천무지체, 구음절맥, 태양화리
- Unique skills, items, spells: 태양의 목걸이, 벨라스틴의 마법진, 플라이 마법
- Monster / demon names: 라이제강, 붉은 달의 라이제강

DO NOT EXTRACT (these are BAD, ignore them):
- Common nouns: 사랑 (love), 시간 (time), 인간 (human), 평화 (peace), 영웅 (hero)
- Abstract concepts: 진실, 의미, 자유, 모험, 모험의 가치
- Generic verbs/adjectives
- Common phrases like "모험의 결과" (result of adventure)

CRITICAL: If a term is just a common Korean word that any dictionary has — SKIP IT.
Only extract words that are SPECIFIC to this novel's universe.

RUSSIAN TRANSLITERATION (Kontsevich system):
- 김 → Ким, 이 → И/Ли, 박 → Пак, 최 → Чхве, 정 → Чон, 강 → Кан
- 유더 바이엘 → Юдер Байель
- 코델리아 체이스 → Корделия Чейз
- 라이제강 → Райзеган
- 마이아 → Майя
- 플레이아데스 → Плеяды
- 세일�␃ 왕국 → королевство Сэйрун
- Do NOT use Chinese pinyin (강진호 is NEVER "Ган Жэньхао", it is "Кан Джинхо")
- No hyphens in given names: "Кан Джинхо", not "Кан Джин-хо"

OUTPUT FORMAT — STRICT JSON ONLY, no commentary, no markdown:
{{
  "terms": [
    {{"korean": "유더 바이엘", "romanization": "Yudeo Baiel", "russian": "Юдер Байель", "category": "name_male", "note": "ГГ, второй сын графа Байеля"}},
    {{"korean": "코델리아 체이스", "romanization": "Kodellia Cheiseu", "russian": "Корделия Чейз", "category": "name_female", "note": "Невеста Юдера, маг"}},
    {{"korean": "플레이아데스", "romanization": "Peulleiadeseu", "russian": "Плеяды", "category": "place", "note": "Мир игры Эпос Героев 2"}}
  ]
}}

Categories: name_male, name_female, place, org, title, term, skill, item, monster

TEXT TO ANALYZE:
{segments_text}

Remember: ONLY proper nouns and unique novel terms. NO common words. Output ONLY the JSON object.
"""
    return call_ollama(prompt)


def normalize_glossary(terms: list, model: str) -> list:
    """Второй проход: модель находит дубликаты, разные романизации одного термина,
    унифицирует переводы."""
    if not terms:
        return terms

    compact = [{"k": t.get("korean"), "r": t.get("russian"),
                "c": t.get("category"), "n": t.get("note", "")[:80]}
               for t in terms]

    prompt = f"""/no_think
You are reviewing a Korean-to-Russian glossary. Find issues:
1. Same Korean term with different Russian translations — pick the best.
2. Wrong romanization (e.g. Chinese pinyin instead of Korean Kontsevich).
3. Inconsistent name styles (e.g. "Кан Джин-хо" and "Кан Джинхо" — unify).
4. Wrong gender of names — note in field "note".

Return the CORRECTED full list in the SAME JSON schema.
Keep ALL unique Korean terms. Only fix the Russian side.

INPUT:
{json.dumps(compact, ensure_ascii=False, indent=2)}

Respond ONLY with JSON:
{{"terms": [{{"korean":"...", "russian":"...", "category":"...", "note":"..."}}]}}
"""
    response = call_ollama(prompt, model=model, num_predict=4096)
    parsed = parse_json_response(response)
    fixed = parsed.get("terms", [])
    if not fixed:
        print("  ! Нормализация не дала результата, возвращаем исходный список")
        return terms

    # Перенесём romanization из исходных (если потерялся)
    rom_map = {t["korean"]: t.get("romanization", "") for t in terms}
    for t in fixed:
        if "romanization" not in t:
            t["romanization"] = rom_map.get(t.get("korean", ""), "")
    return fixed


def parse_json_response(response: str) -> dict:
    # Убираем think-теги если есть
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    array_match = re.search(r'\[[\s\S]*\]', response)
    if array_match:
        try:
            return {"terms": json.loads(array_match.group())}
        except json.JSONDecodeError:
            pass
    return {"terms": []}


def merge_glossaries(glossary_parts: list) -> list:
    merged = {}
    for part in glossary_parts:
        if not isinstance(part, dict):
            continue
        terms = part.get("terms", [])
        if not isinstance(terms, list):
            continue
        for term in terms:
            # ГЛАВНОЕ: фильтруем не-словари
            if not isinstance(term, dict):
                continue
            korean = (term.get("korean") or "").strip()
            if not korean:
                continue
            if korean not in merged:
                merged[korean] = term
            else:
                if len(term.get("note", "") or "") > len(merged[korean].get("note", "") or ""):
                    merged[korean] = term
    return list(merged.values())


def main():
    parser = argparse.ArgumentParser(description='Этап 2: Создание глоссария')
    parser.add_argument('--input', type=str, default=SEGMENTS_FILE)
    parser.add_argument('--output', type=str, default=GLOSSARY_FILE)
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Сколько сегментов анализировать (0 = все)')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--model', type=str, default=MODEL_GLOSSARY)
    parser.add_argument('--no-normalize', action='store_true',
                        help='Пропустить второй проход нормализации')
    args = parser.parse_args()
    CHECKPOINT_FILE = args.output + ".checkpoint.jsonl"

    if not os.path.exists(args.input):
        print(f"ОШИБКА: Файл не найден: {args.input}")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = [s["text"] for s in data["segments"]]

    if args.sample_size > 0:
        step = max(1, len(segments) // args.sample_size)
        segments = segments[::step][:args.sample_size]

    print(f"Анализ {len(segments)} сегментов")
    print(f"Модель: {args.model}, батч: {args.batch_size}")
    print("-" * 50)

    glossary_parts = []
    total_batches = (len(segments) + args.batch_size - 1) // args.batch_size

    done_batches = set()
    glossary_parts = []
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_batches.add(rec["batch_num"])
                    glossary_parts.append(rec["data"])
                except Exception:
                    pass
        print(f"Восстановлено {len(done_batches)} батчей из чекпойнта")

    total_batches = (len(segments) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(segments), args.batch_size):
        batch_num = i // args.batch_size + 1
        if batch_num in done_batches:
            continue

        batch = segments[i:i + args.batch_size]
        print(f"  Батч {batch_num}/{total_batches}...", end=" ", flush=True)

        batch_text = "\n\n---\n\n".join(batch)
        response = extract_terms_from_batch(batch_text)

        parsed = {"terms": []}
        if response:
            try:
                parsed = parse_json_response(response)
            except Exception as e:
                print(f"парс-ошибка: {e}", end=" ")

        # Валидация: оставляем только словари
        if isinstance(parsed, dict):
            raw = parsed.get("terms", [])
            if isinstance(raw, list):
                parsed["terms"] = [t for t in raw if isinstance(t, dict)]
            else:
                parsed = {"terms": []}
        else:
            parsed = {"terms": []}

        glossary_parts.append(parsed)
        print(f"найдено {len(parsed['terms'])} терминов")

        # СОХРАНЯЕМ СРАЗУ
        with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"batch_num": batch_num, "data": parsed},
                               ensure_ascii=False) + "\n")

        time.sleep(1)

    merged_terms = merge_glossaries(glossary_parts)
    print(f"\nПосле объединения: {len(merged_terms)} уникальных терминов")

    if not args.no_normalize and len(merged_terms) > 0:
        print("Нормализация (второй проход)...")
        merged_terms = normalize_glossary(merged_terms, args.model)
        print(f"После нормализации: {len(merged_terms)} терминов")

    merged_terms.sort(key=lambda x: (x.get("category", ""), x.get("korean", "")))

    output_data = {
        "metadata": {
            "model": args.model,
            "segments_analyzed": len(segments),
            "total_terms": len(merged_terms),
            "normalized": not args.no_normalize,
            "categories": {}
        },
        "terms": merged_terms
    }
    for term in merged_terms:
        cat = term.get("category", "unknown")
        output_data["metadata"]["categories"][cat] = \
            output_data["metadata"]["categories"].get(cat, 0) + 1

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено: {args.output}")
    for cat, count in output_data["metadata"]["categories"].items():
        print(f"  {cat}: {count}")
    print("\nПревью (первые 15):")
    for term in merged_terms[:15]:
        print(f"  {term.get('korean')} → {term.get('russian')} "
              f"[{term.get('category', '')}] {term.get('note', '')}")
    print("\n⚠ ВАЖНО: Проверьте глоссарий вручную перед этапом 3!")


if __name__ == "__main__":
    main()
