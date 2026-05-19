import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta

import requests

from config import *


# ============================================================
# КОНСТАНТЫ
# ============================================================

# Максимум терминов глоссария на один промпт (чтобы не раздувать контекст)
MAX_GLOSSARY_TERMS = 60

# Приоритет категорий при отборе (если терминов больше лимита)
CATEGORY_PRIORITY = [
    "name_male", "name_female", "name",
    "place", "org", "title",
    "skill", "monster", "item",
    "term", "other",
]

CATEGORY_TITLES_RU = {
    "name_male":   "Мужские имена",
    "name_female": "Женские имена",
    "name":        "Имена",
    "place":       "Места",
    "org":         "Организации",
    "title":       "Титулы и звания",
    "skill":       "Навыки и техники",
    "monster":     "Существа",
    "item":        "Предметы",
    "term":        "Термины мира",
    "other":       "Прочее",
}


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(
    prompt: str,
    model: str,
    temperature: float,
    num_predict: int = 4096,
    num_ctx: int = 8192,
    keep_alive: str = "30m",
    max_retries: int = 3,
) -> str:
    """Запрос к Ollama с retry и keep_alive (модель не выгружается из VRAM)."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            # >>> Рекомендованные Qwen3 параметры для non-thinking режима:
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.05,
        },
    }

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=1200)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            print("\nОШИБКА: Ollama недоступна. Запустите: ollama serve")
            sys.exit(1)
        except requests.exceptions.Timeout:
            last_err = "timeout"
            print(f"\n  ! Timeout (попытка {attempt}/{max_retries}), пауза 15с...")
            time.sleep(15)
        except Exception as e:
            last_err = str(e)
            print(f"\n  ! Ошибка: {e} (попытка {attempt}/{max_retries})")
            time.sleep(5)

    print(f"  ! Не удалось получить ответ за {max_retries} попыток ({last_err})")
    return ""


# ============================================================
# ГЛОССАРИЙ
# ============================================================

def find_relevant_terms(glossary_terms: list, segment_text: str,
                        max_terms: int = MAX_GLOSSARY_TERMS) -> list:
    """
    Найти термины из глоссария, реально встречающиеся в сегменте.
    Сортирует по частоте вхождения и длине термина (длинные приоритетнее,
    чтобы 유더 바이엘 был выше, чем просто 유더).
    """
    matched = []
    for t in glossary_terms:
        kor = (t.get("korean") or "").strip()
        if not kor:
            continue
        count = segment_text.count(kor)
        if count > 0:
            matched.append((count, len(kor), t))

    # Сортировка: больше вхождений → длиннее термин
    matched.sort(key=lambda x: (-x[0], -x[1]))

    if len(matched) <= max_terms:
        return [m[2] for m in matched]

    # Если терминов слишком много — сначала приоритетные категории
    cat_order = {c: i for i, c in enumerate(CATEGORY_PRIORITY)}

    def sort_key(item):
        count, length, term = item
        cat = term.get("category", "other")
        return (cat_order.get(cat, 99), -count, -length)

    matched.sort(key=sort_key)
    return [m[2] for m in matched[:max_terms]]


def format_glossary_for_prompt(relevant_terms: list) -> str:
    """Группировка по категориям с человекочитаемыми заголовками."""
    if not relevant_terms:
        return "(в этом сегменте нет специфических терминов из глоссария)"

    by_cat = {}
    for t in relevant_terms:
        cat = t.get("category") or "other"
        by_cat.setdefault(cat, []).append(t)

    lines = []
    for cat in CATEGORY_PRIORITY:
        if cat not in by_cat:
            continue
        lines.append(f"  [{CATEGORY_TITLES_RU.get(cat, cat)}]")
        for t in by_cat[cat]:
            line = f"    {t.get('korean')} → {t.get('russian')}"
            note = (t.get("note") or "").strip()
            if note:
                line += f"  ({note})"
            lines.append(line)

    # Категории не из списка — в конец
    extras = [c for c in by_cat if c not in CATEGORY_PRIORITY]
    if extras:
        for cat in extras:
            lines.append(f"  [{cat}]")
            for t in by_cat[cat]:
                lines.append(f"    {t.get('korean')} → {t.get('russian')}")

    return "\n".join(lines)


# ============================================================
# ПЕРЕВОД
# ============================================================

def clean_translation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)  # незакрытый think
    text = re.sub(r"^```[\w]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Расширенный список префиксов
    prefixes = [
        r"^(RUSSIAN TRANSLATION|Перевод|Translation|РУССКИЙ ПЕРЕВОД"
        r"|Russian|Перевод на русский|Here is the translation"
        r"|Here's the translation|Конечно[,!]?\s*вот перевод"
        r"|Вот перевод)[:：]?\s*\n?",
        r"^\*\*[^*]+\*\*[:：]?\s*\n",   # **Перевод:**
        r"^---\s*\n",
    ]
    for p in prefixes:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n---\s*$", "", text)
    return text.strip()


def validate_translation(translation: str, source: str) -> tuple:
    """Базовая валидация — возвращает (ok, reason)."""
    if not translation:
        return False, "пустой ответ"
    if len(translation) < 10:
        return False, f"слишком короткий ({len(translation)} симв)"

    korean_chars = sum(1 for c in translation if "\uac00" <= c <= "\ud7af")
    cyrillic_chars = sum(1 for c in translation if "\u0400" <= c <= "\u04ff")

    if korean_chars > len(translation) * 0.20:
        return False, f"много корейских символов ({korean_chars}/{len(translation)})"
    if cyrillic_chars < len(translation) * 0.30:
        return False, f"мало кириллицы ({cyrillic_chars}/{len(translation)})"

    # Длина перевода не должна быть < 30% от исходника
    if len(translation) < len(source) * 0.30:
        return False, f"перевод сильно короче исходника"

    return True, ""


def translate_segment(
    segment_text: str,
    glossary_terms: list,
    context_before: str,
    segment_id: int,
    model: str,
    temperature: float,
    max_glossary: int,
) -> str:
    relevant = find_relevant_terms(glossary_terms, segment_text, max_terms=max_glossary)
    glossary_section = format_glossary_for_prompt(relevant)

    context_section = ""
    if context_before:
        # Берём последние 1500 символов предыдущего перевода
        ctx = context_before[-1500:]
        context_section = (
            "\nПРЕДЫДУЩИЙ ПЕРЕВОД (только для связности стиля — НЕ переводить и НЕ повторять):\n"
            f"---\n{ctx}\n---\n"
        )

    prompt = f"""/no_think
Ты — профессиональный литературный переводчик с корейского на русский, специализируешься на корейских ранобэ и веб-новеллах.
Переведи приведённый корейский текст на естественный, живой русский язык — так, как написал бы хороший русскоязычный автор.

ПРАВИЛА:
1. Точный смысл, но русский синтаксис и ритм. Не калькируй корейский порядок слов.
2. Сохраняй стиль и тон оригинала: сленг → русским сленгом, формальное → формально, грубое → грубо.
3. ОБЯЗАТЕЛЬНО используй переводы из глоссария ниже — имена/термины должны быть единообразными.
4. Имена склоняй естественно по русским правилам:
   - Юдер → Юдера, Юдеру, с Юдером
   - Корделия → Корделии, Корделию
   - Хон Юхи, Чхвирён — несклоняемые корейские (как «Окубо»)
5. Сохраняй разбиение на абзацы и диалоги (каждая реплика с новой строки).
6. Никаких пояснений, комментариев, заголовков, рассуждений — ТОЛЬКО перевод.
7. Не добавляй ничего, чего нет в оригинале. Не упрощай и не сокращай.
8. Корейский игровой/сетевой сленг переводи русскими аналогами:
   - 고인물 / 썩은물 → «олд», «прошарик», «задрот» (по контексту)
   - 닥쳐 → «заткнись»
   - 야 → «эй», «слушай»
   - 헐 / 헉 → «офигеть», «ого»
   - ㅋㅋㅋ → «ахах», «лол» (по контексту)

ГЛОССАРИЙ (используй ровно эти переводы; склоняй по русским правилам):
{glossary_section}
{context_section}
КОРЕЙСКИЙ ТЕКСТ (Сегмент #{segment_id}):
---
{segment_text}
---

ПЕРЕВОД (только русский текст, без префиксов и комментариев):"""

    return call_ollama(prompt, model=model, temperature=temperature,
                       num_predict=4096, num_ctx=8192)


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def atomic_save(output_data: dict, output_path: str):
    """Сохранение через временный файл — на случай падения посреди записи."""
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)  # атомарно на всех ОС


def build_output(translations: dict, total_segments: int,
                 model: str, glossary_count: int) -> dict:
    return {
        "metadata": {
            "model": model,
            "glossary_terms_count": glossary_count,
            "translated_segments": sum(1 for t in translations.values() if not t.get("error")),
            "total_segments": total_segments,
            "errors": sum(1 for t in translations.values() if t.get("error")),
            "last_updated": datetime.now().isoformat(),
        },
        "translations": sorted(translations.values(), key=lambda x: x["id"]),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Этап 3: Перевод с глоссарием")
    parser.add_argument("--segments", type=str, default=SEGMENTS_FILE)
    parser.add_argument("--glossary", type=str, default=GLOSSARY_FILE)
    parser.add_argument("--output", type=str, default=TRANSLATED_FILE)
    parser.add_argument("--start", type=int, default=1, help="ID первого сегмента")
    parser.add_argument("--end", type=int, default=0, help="ID последнего (0 = все)")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить, пропуская уже переведённые без ошибок")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Перевести заново только сегменты с ошибками")
    parser.add_argument("--model", type=str, default=MODEL_TRANSLATE)
    parser.add_argument("--temperature", type=float, default=TRANSLATION_TEMPERATURE)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Пауза между сегментами (сек)")
    parser.add_argument("--max-glossary", type=int, default=MAX_GLOSSARY_TERMS,
                        help="Лимит терминов глоссария в одном промпте")
    parser.add_argument("--no-backup", action="store_true",
                        help="Не делать бэкап существующего output")
    args = parser.parse_args()

    # --- Загрузка сегментов ---
    if not os.path.exists(args.segments):
        print(f"ОШИБКА: Файл не найден: {args.segments}")
        sys.exit(1)

    with open(args.segments, "r", encoding="utf-8") as f:
        segments_data = json.load(f)
    segments = segments_data["segments"]
    segments_by_id = {s["id"]: s for s in segments}

    # --- Загрузка глоссария ---
    glossary_terms = []
    if os.path.exists(args.glossary):
        with open(args.glossary, "r", encoding="utf-8") as f:
            glossary_terms = json.load(f).get("terms", [])
        print(f"Глоссарий: {len(glossary_terms)} терминов")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: Глоссарий не найден — переводим без него")

    # --- Загрузка существующих переводов ---
    existing = {}
    if (args.resume or args.retry_errors) and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("translations", []):
                existing[item["id"]] = item
        print(f"Загружено {len(existing)} существующих переводов")

        # Бэкап
        if not args.no_backup:
            backup = args.output + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(args.output, backup)
            print(f"Бэкап: {backup}")

    # --- Что переводить ---
    start_id = args.start
    end_id = args.end if args.end > 0 else (max(s["id"] for s in segments))

    if args.retry_errors:
        to_translate_ids = [
            sid for sid, t in existing.items()
            if t.get("error") and start_id <= sid <= end_id
        ]
        to_translate_ids.sort()
        print(f"Режим RETRY-ERRORS: будет переведено заново {len(to_translate_ids)} ошибочных")
    elif args.resume:
        good_ids = {sid for sid, t in existing.items() if not t.get("error")}
        to_translate_ids = [
            s["id"] for s in segments
            if start_id <= s["id"] <= end_id and s["id"] not in good_ids
        ]
        skipped = sum(1 for s in segments
                      if start_id <= s["id"] <= end_id and s["id"] in good_ids)
        if skipped:
            print(f"Пропущено уже переведённых: {skipped}")
    else:
        to_translate_ids = [s["id"] for s in segments
                            if start_id <= s["id"] <= end_id]

    if not to_translate_ids:
        print("Нечего переводить.")
        return

    print(f"\nК переводу: {len(to_translate_ids)} сегментов")
    print(f"Модель: {args.model} | температура: {args.temperature}")
    print(f"Лимит глоссария на промпт: {args.max_glossary}")
    print("-" * 60)

    translations = dict(existing)
    run_start = time.time()
    processed_ok = 0
    processed_fail = 0

    for idx, seg_id in enumerate(to_translate_ids):
        segment = segments_by_id[seg_id]
        seg_text = segment["text"]
        char_count = segment.get("char_count", len(seg_text))

        # Контекст из предыдущего сегмента (если он переведён успешно)
        prev_id = seg_id - 1
        context_before = ""
        if prev_id in translations and not translations[prev_id].get("error"):
            context_before = translations[prev_id].get("translation", "")

        # ETA
        if processed_ok + processed_fail > 0:
            elapsed = time.time() - run_start
            avg = elapsed / (processed_ok + processed_fail)
            remaining = (len(to_translate_ids) - idx) * avg
            eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M %d.%m")
            eta_str = f" | ETA {eta} ({remaining/3600:.1f}ч)"
        else:
            eta_str = ""

        # Сколько терминов глоссария реально подойдёт
        relevant_count = len(find_relevant_terms(glossary_terms, seg_text,
                                                 max_terms=args.max_glossary))

        print(f"[{idx+1}/{len(to_translate_ids)}] #{seg_id} "
              f"({char_count} симв | глос: {relevant_count}){eta_str}",
              flush=True)

        start_time = time.time()
        raw = translate_segment(
            seg_text, glossary_terms, context_before, seg_id,
            args.model, args.temperature, args.max_glossary,
        )
        elapsed = time.time() - start_time

        translation = clean_translation(raw) if raw else ""
        ok, reason = validate_translation(translation, seg_text)

        if ok:
            translations[seg_id] = {
                "id": seg_id,
                "source": seg_text,
                "translation": translation,
                "model": args.model,
                "time_seconds": round(elapsed, 1),
            }
            processed_ok += 1
            preview = translation[:90].replace("\n", " ")
            if len(translation) > 90:
                preview += "..."
            print(f"  OK ({elapsed:.0f}с, {len(translation)} симв)")
            print(f"  → {preview}")
        else:
            translations[seg_id] = {
                "id": seg_id,
                "source": seg_text,
                "translation": translation or "[ОШИБКА ПЕРЕВОДА]",
                "model": args.model,
                "error": True,
                "error_reason": reason,
                "time_seconds": round(elapsed, 1),
            }
            processed_fail += 1
            print(f"  FAIL: {reason}")

        # Сохраняем после каждого сегмента (атомарно)
        try:
            atomic_save(
                build_output(translations, len(segments), args.model, len(glossary_terms)),
                args.output,
            )
        except Exception as e:
            print(f"  !! Ошибка сохранения: {e}")

        if idx < len(to_translate_ids) - 1:
            time.sleep(args.delay)

    # --- Итоги ---
    total_time = time.time() - run_start
    print("\n" + "=" * 60)
    print(f"Готово за {total_time/60:.1f} минут")
    print(f"Успешно: {processed_ok}, Ошибок: {processed_fail}")
    print(f"Файл: {args.output}")
    if processed_fail > 0:
        print("\nДля повторной попытки ошибочных:")
        print(f"  python {os.path.basename(sys.argv[0])} --retry-errors")


if __name__ == "__main__":
    main()
