"""
Этап 2: Извлечение глоссария из корейского текста (по умолчанию qwen).

Что нового против ver1.0:
  - ПОЛЕ hanja (漢字). Для сино-корейских концепт-терминов модель сперва
    восстанавливает иероглифы, а перевод делает ПО ИХ СМЫСЛУ, а не фонетикой.
    Это лечит «Гуин Цзюймэй»/«Чхонму Чхи» у корня (исследование: ханча снимает
    семантическую неоднозначность корейских омонимов в LLM).
  - Посев ГОНОРАТИВОВ (config.HONORIFICS) как builtin-терминов. Экстрактор
    намеренно выбрасывает «общие слова», поэтому обращения вроде 도련님 раньше
    не попадали в глоссарий и угадывались фонетически («Домини»). Теперь они
    всегда в глоссарии и не правятся кросс-проверкой (stage3).

Запуск:  python stage2_glossary.py
         python stage2_glossary.py --no-normalize
"""

import json
import re
import argparse
import sys
import requests
import time
import os
from config import *


def call_ollama(system: str, user: str, model: str = MODEL_GLOSSARY,
                temperature: float = GLOSSARY_TEMPERATURE,
                num_predict: int = 4096,
                num_ctx: int = GLOSSARY_NUM_CTX) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # ВАЖНО: num_ctx обязателен — иначе Ollama берёт дефолт 2048
        # и батч из нескольких сегментов молча обрезается.
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=1200)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        print("ОШИБКА: Не удаётся подключиться к Ollama. Запустите: ollama serve")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        body = getattr(e.response, "text", "")
        print(f"ОШИБКА HTTP: {e}\n  Ответ Ollama: {body}")
        print("  Подсказка: 404 обычно = модель не найдена. Сверьте имя в config с `ollama list`.")
        return ""
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return ""


def extract_terms_from_batch(segments_text: str) -> str:
    system = f"""You are extracting a translation glossary from a Korean light novel.

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
- Honorifics / forms of address (도련님, 아가씨, 폐하 ...) — these are injected separately, SKIP them.

CRITICAL: If a term is just a common Korean word that any dictionary has — SKIP IT.
Only extract words that are SPECIFIC to this novel's universe.

==================================================================
HANJA FIELD (漢字) — THIS IS THE MOST IMPORTANT RULE FOR CONCEPT TERMS:
Many martial-arts / fantasy concept terms are Sino-Korean (한자어). For EVERY such
term you MUST:
  1. Recover the underlying Hanja into the "hanja" field (e.g. 구음절맥 → 九陰絶脈,
     천무지체 → 天武之體, 변경백 → 邊境伯).
  2. Translate into Russian BY THE MEANING OF THOSE CHARACTERS — never a bare
     phonetic transliteration of the Korean reading.
     RIGHT: 구음절맥 (九陰絶脈) → «меридианы Девяти Инь» / «прерванные меридианы Девяти Инь»
     WRONG: 구음절맥 → «Гуин Цзюймэй»   (phonetic — FORBIDDEN)
     RIGHT: 천무지체 (天武之體) → «тело Небесного Воина» / «небесное боевое тело»
     WRONG: 천무지체 → «Чхонму Чхи»     (phonetic — FORBIDDEN)
For pure NAMES / PLACES with no meaningful Hanja, leave "hanja" empty ("") and
transliterate by Kontsevich (see below). Never invent Hanja for native Korean names.
==================================================================

RUSSIAN TRANSLITERATION (Kontsevich system, for names/places only):
- 김 → Ким, 이 → И/Ли, 박 → Пак, 최 → Чхве, 정 → Чон, 강 → Кан
- 유더 바이엘 → Юдер Байель
- 코델리아 체이스 → Корделия Чейз
- 라이제강 → Райзеган
- 마이아 → Майя
- 플레이아데스 → Плеяды
- 세일룬 왕국 → королевство Сэйрун
- Do NOT use Chinese pinyin (강진호 is NEVER "Ган Жэньхао", it is "Кан Джинхо")
- No hyphens in given names: "Кан Джинхо", not "Кан Джин-хо"

DISAMBIGUATION RULES (apply to EVERY term — this is where mistakes happen):
- FOREIGN LOANWORDS (외래어): many terms are Korean transliterations of English/other
  foreign words. Recover the ORIGINAL word, do NOT transliterate blindly:
  플라이 → Fly (магия полёта), 아웃복서 → Outboxer (Аутбоксер), 프로스트 앤빌 → Frost Anvil (Фрост Анвил),
  카플란 효과 → эффект Каплана (Kaplan). Note: ㅍ часто = F (не «П»), ㄹ = L (не «Р»).
- HANJA HOMOPHONES: у корейских слов много омонимов — выбирай значение ПО КОНТЕКСТУ соседних
  сегментов. Напр.: 변경 = 邊境 (граница → «маркграф»), НЕ 變更 (перемена). 성 = 聖 (святой).
- NUMBERS: цифры держи точно — 구(9), 십(10), 삼(3), 천(тысяча). Никогда не путай 9 и 10.
- ТИТУЛЫ (фикс. перевод): 공작=герцог, 대공=великий герцог, 후작=маркиз, 변경백=маркграф,
  백작=граф, 자작=виконт, 남작=барон; 백작가=«графский род / дом графа» (НЕ барон).

OUTPUT FORMAT — STRICT JSON ONLY, no commentary, no markdown:
{{
  "terms": [
    {{"korean": "유더 바이엘", "hanja": "", "romanization": "Yudeo Baiel", "russian": "Юдер Байель", "category": "name_male", "note": "ГГ, второй сын графа Байеля"}},
    {{"korean": "구음절맥", "hanja": "九陰絶脈", "romanization": "Gueumjeolmaek", "russian": "меридианы Девяти Инь", "category": "term", "note": "врождённая блокировка каналов ци"}},
    {{"korean": "천무지체", "hanja": "天武之體", "romanization": "Cheonmujiche", "russian": "тело Небесного Воина", "category": "term", "note": "врождённое боевое тело"}},
    {{"korean": "플레이아데스", "hanja": "", "romanization": "Peulleiadeseu", "russian": "Плеяды", "category": "place", "note": "Мир игры Эпос Героев 2"}}
  ]
}}

Categories: name_male, name_female, place, org, title, term, skill, item, monster

Remember: ONLY proper nouns and unique novel terms. NO common words. NO honorifics.
Fill "hanja" for every Sino-Korean concept term. Output ONLY the JSON object.
"""
    user = f"/no_think\nTEXT TO ANALYZE:\n\n{segments_text}"
    return call_ollama(system, user)


def _normalize_chunk(compact: list, model: str) -> list:
    """Нормализация ОДНОГО чанка терминов."""
    system = f"""You are reviewing a Korean-to-Russian glossary. Find issues:
1. Same Korean term with different Russian translations — pick the best.
2. Wrong romanization (e.g. Chinese pinyin instead of Korean Kontsevich).
3. Inconsistent name styles (e.g. "Кан Джин-хо" and "Кан Джинхо" — unify).
4. Wrong gender of names — note in field "note".
5. Foreign loanwords transliterated blindly — recover the original word
   (플라이→Fly, 아웃복서→Outboxer, 프로스트 앤빌→Frost Anvil, 카플란→Kaplan; ㅍ=F, ㄹ=L).
6. Hanja homophones misread (변경 граница→«маркграф» vs перемена); wrong numbers (구=9, не 10).
7. CONCEPT TERMS transliterated phonetically instead of translated by Hanja meaning.
   If field "h" (hanja) is present, translate the Russian by the MEANING of those
   characters, NOT by the Korean sound:
     九陰絶脈 → «меридианы Девяти Инь» (НЕ «Гуин Цзюймэй»)
     天武之體 → «тело Небесного Воина» (НЕ «Чхонму Чхи»)

Return the CORRECTED full list in the SAME JSON schema.
Keep ALL unique Korean terms. Do NOT change the "k" (korean) or "h" (hanja) fields.
Only fix the Russian side and category/note.

Respond ONLY with JSON:
{{"terms": [{{"korean":"...", "russian":"...", "category":"...", "note":"..."}}]}}
"""
    user = f"/no_think\nINPUT:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    response = call_ollama(system, user, model=model, num_predict=4096)
    parsed = parse_json_response(response)
    return parsed.get("terms", [])


def normalize_glossary(terms: list, model: str,
                       chunk_size: int = GLOSSARY_NORMALIZE_CHUNK) -> list:
    """Второй проход: дубликаты, разные романизации, унификация переводов.
    Разбиваем на чанки, чтобы не переполнять контекст на больших глоссариях.
    korean и hanja НЕ меняются — они источник истины и восстанавливаются по карте."""
    if not terms:
        return terms

    rom_map = {t.get("korean"): t.get("romanization", "") for t in terms}
    hanja_map = {t.get("korean"): t.get("hanja", "") for t in terms}
    fixed_all = []

    total_chunks = (len(terms) + chunk_size - 1) // chunk_size
    for ci in range(0, len(terms), chunk_size):
        chunk = terms[ci:ci + chunk_size]
        compact = [{"k": t.get("korean"), "h": t.get("hanja", ""),
                    "r": t.get("russian"), "c": t.get("category"),
                    "n": (t.get("note", "") or "")[:80]}
                   for t in chunk]
        chunk_num = ci // chunk_size + 1
        print(f"  Нормализация чанка {chunk_num}/{total_chunks} "
              f"({len(chunk)} терминов)...", end=" ", flush=True)

        fixed = _normalize_chunk(compact, model)
        if not fixed:
            print("без изменений (оставляем исходные)")
            fixed_all.extend(chunk)
            continue

        # Приводим компактный формат обратно + восстанавливаем romanization/hanja по карте
        restored = []
        for t in fixed:
            kor = t.get("korean") or t.get("k") or ""
            restored.append({
                "korean": kor,
                "hanja": hanja_map.get(kor, t.get("hanja") or t.get("h") or ""),
                "russian": t.get("russian") or t.get("r") or "",
                "category": t.get("category") or t.get("c") or "",
                "note": t.get("note") or t.get("n") or "",
                "romanization": t.get("romanization") or rom_map.get(kor, ""),
            })
        print(f"ок ({len(restored)})")
        fixed_all.extend(restored)
        time.sleep(1)

    # Подстраховка: если по какой-то причине потеряли термины — добираем
    seen = {t.get("korean") for t in fixed_all}
    for t in terms:
        if t.get("korean") not in seen:
            fixed_all.append(t)

    return fixed_all


def seed_honorifics(terms: list) -> list:
    """Принудительно добавляет гоноративы/обращения (config.HONORIFICS) как
    builtin-термины, если их корейский ключ ещё не присутствует. builtin=True
    защищает их от правок на кросс-проверке (stage3)."""
    present = {(t.get("korean") or "").strip() for t in terms}
    added = 0
    for h in HONORIFICS:
        k = (h.get("korean") or "").strip()
        if not k or k in present:
            continue
        terms.append({
            "korean": k,
            "hanja": h.get("hanja", ""),
            "romanization": h.get("romanization", ""),
            "russian": h.get("russian", ""),
            "category": HONORIFICS_CATEGORY,
            "note": h.get("note", ""),
            "builtin": True,
        })
        present.add(k)
        added += 1
    if added:
        print(f"Посеяно гоноративов (builtin): {added}")
    return terms


def parse_json_response(response: str) -> dict:
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
            if not isinstance(term, dict):
                continue
            korean = (term.get("korean") or "").strip()
            if not korean:
                continue
            if korean not in merged:
                merged[korean] = term
            else:
                # при дубле берём запись с более длинной заметкой,
                # но не теряем уже найденную hanja
                old = merged[korean]
                if len(term.get("note", "") or "") > len(old.get("note", "") or ""):
                    if not term.get("hanja") and old.get("hanja"):
                        term["hanja"] = old["hanja"]
                    merged[korean] = term
                elif not old.get("hanja") and term.get("hanja"):
                    old["hanja"] = term["hanja"]
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
    parser.add_argument('--no-honorifics', action='store_true',
                        help='Не сеять гоноративы из config.HONORIFICS')
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
    print(f"Модель: {args.model}, батч: {args.batch_size}, num_ctx: {GLOSSARY_NUM_CTX}")
    print("-" * 50)

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

        with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"batch_num": batch_num, "data": parsed},
                               ensure_ascii=False) + "\n")

        time.sleep(1)

    merged_terms = merge_glossaries(glossary_parts)
    print(f"\nПосле объединения: {len(merged_terms)} уникальных терминов")

    if not args.no_normalize and len(merged_terms) > 0:
        print("Нормализация (второй проход, по чанкам)...")
        merged_terms = normalize_glossary(merged_terms, args.model)
        print(f"После нормализации: {len(merged_terms)} терминов")

    # Гоноративы сеем ПОСЛЕ нормализации — чтобы их не «причесали»
    if not args.no_honorifics:
        merged_terms = seed_honorifics(merged_terms)

    # Гарантируем наличие полей hanja/romanization у всех терминов
    for t in merged_terms:
        t.setdefault("hanja", "")
        t.setdefault("romanization", "")

    merged_terms.sort(key=lambda x: (x.get("category", ""), x.get("korean", "")))

    output_data = {
        "metadata": {
            "model": args.model,
            "segments_analyzed": len(segments),
            "total_terms": len(merged_terms),
            "normalized": not args.no_normalize,
            "honorifics_seeded": not args.no_honorifics,
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
        hj = f" [{term['hanja']}]" if term.get("hanja") else ""
        print(f"  {term.get('korean')}{hj} → {term.get('russian')} "
              f"[{term.get('category', '')}] {term.get('note', '')}")
    print("\n⚠ ВАЖНО: Проверьте глоссарий вручную перед этапом 3 (кросс-проверка)!")
    print("Дальше: python stage3_glossary_check.py")


if __name__ == "__main__":
    main()
