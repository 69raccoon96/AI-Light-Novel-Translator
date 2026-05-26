"""
Этап 4: Перевод сегментов с глоссарием (по умолчанию qwen).

Что нового против ver1.0:
  - Сырой «ПРЕДЫДУЩИЙ ПЕРЕВОД» в промпте ВЫКЛЮЧЕН по умолчанию
    (config.TRANSLATE_USE_PREV_CONTEXT=False). Именно он провоцировал
    галлюцинации-копии (в ver1.0 в #25 продублировался диалог из #24).
    Связность держим абстрактным «бегущим саммари» (story bible).
  - Глоссарий в промпте показывает HANJA: «KO (漢字) → RU (note)» — модель
    переводит концепт-термины по смыслу, а не фонетикой.
  - Параметры декодирования вынесены в config (top_p/top_k/penalty).

Запуск:  python stage4_translate.py
         python stage4_translate.py --resume
         python stage4_translate.py --retry-errors
"""

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

MAX_GLOSSARY_TERMS = 60

CATEGORY_PRIORITY = [
    "name_male", "name_female", "name",
    "place", "org", "title", "address",
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
    "address":     "Обращения",
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
    system: str,
    user: str,
    model: str,
    temperature: float,
    num_predict: int = 4096,
    num_ctx: int = TRANSLATE_NUM_CTX,
    keep_alive: str = "30m",
    max_retries: int = 3,
) -> str:
    """Запрос к Ollama (/api/chat): system — статичные правила, user — данные."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    options = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        # Рекомендованные Qwen3 параметры для non-thinking режима:
        "top_p": TRANSLATE_TOP_P,
        "top_k": TRANSLATE_TOP_K,
        "repeat_penalty": TRANSLATE_REPEAT_PENALTY,
    }
    if TRANSLATE_PRESENCE_PENALTY and TRANSLATE_PRESENCE_PENALTY > 0:
        options["presence_penalty"] = TRANSLATE_PRESENCE_PENALTY
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
    }

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=1200)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            print("\nОШИБКА: Ollama недоступна. Запустите: ollama serve")
            sys.exit(1)
        except requests.exceptions.Timeout:
            last_err = "timeout"
            print(f"\n  ! Timeout (попытка {attempt}/{max_retries}), пауза 15с...")
            time.sleep(15)
        except requests.exceptions.HTTPError as e:
            body = getattr(e.response, "text", "")
            print(f"\nОШИБКА HTTP: {e}\n  Ответ Ollama: {body}")
            print("  Подсказка: 404 обычно = модель не найдена. Сверьте имя в config с `ollama list`.")
            return ""
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
    """Термины из глоссария, реально встречающиеся в сегменте.
    Сортирует по частоте и длине (длинные приоритетнее)."""
    matched = []
    for t in glossary_terms:
        kor = (t.get("korean") or "").strip()
        if not kor:
            continue
        count = segment_text.count(kor)
        if count > 0:
            matched.append((count, len(kor), t))

    matched.sort(key=lambda x: (-x[0], -x[1]))

    if len(matched) <= max_terms:
        return [m[2] for m in matched]

    cat_order = {c: i for i, c in enumerate(CATEGORY_PRIORITY)}

    def sort_key(item):
        count, length, term = item
        cat = term.get("category", "other")
        return (cat_order.get(cat, 99), -count, -length)

    matched.sort(key=sort_key)
    return [m[2] for m in matched[:max_terms]]


def format_glossary_for_prompt(relevant_terms: list) -> str:
    """Группировка по категориям; показываем hanja для концепт-терминов."""
    if not relevant_terms:
        return "(в этом сегменте нет специфических терминов из глоссария)"

    by_cat = {}
    for t in relevant_terms:
        cat = t.get("category") or "other"
        by_cat.setdefault(cat, []).append(t)

    def fmt_term(t):
        kor = t.get("korean")
        hanja = (t.get("hanja") or "").strip()
        head = f"{kor} ({hanja})" if hanja else f"{kor}"
        line = f"    {head} → {t.get('russian')}"
        note = (t.get("note") or "").strip()
        if note:
            line += f"  ({note})"
        return line

    lines = []
    for cat in CATEGORY_PRIORITY:
        if cat not in by_cat:
            continue
        lines.append(f"  [{CATEGORY_TITLES_RU.get(cat, cat)}]")
        for t in by_cat[cat]:
            lines.append(fmt_term(t))

    extras = [c for c in by_cat if c not in CATEGORY_PRIORITY]
    for cat in extras:
        lines.append(f"  [{cat}]")
        for t in by_cat[cat]:
            lines.append(fmt_term(t))

    return "\n".join(lines)


# ============================================================
# БЕГУЩЕЕ САММАРИ (story bible) — сквозная связность
# ============================================================

def _strip_model_noise(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"```\w*\n?", "", text)
    text = text.replace("```", "")
    return text.strip()


def update_running_summary(prev_summary: str, recent_texts: list,
                           model: str,
                           max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """Обновляет короткую «памятку переводчика» по последним переводам.
    При любой ошибке возвращает прежнюю памятку (не ломает перевод)."""
    joined = "\n\n".join(t for t in recent_texts if t).strip()
    if not joined:
        return prev_summary
    joined = joined[-4000:]

    system = (
        "Ты ведёшь краткую «памятку переводчика» (story bible) для книги — чтобы дальше\n"
        "держать единообразие имён, рода персонажей, тона и обращений.\n\n"
        "Обнови памятку с учётом нового фрагмента перевода. Памятка должна быть СЖАТОЙ\n"
        "и содержать ТОЛЬКО то, что важно для связности дальше:\n"
        "- ключевые персонажи: имя (ровно как в переводе) + пол (м/ж) + кратко роль/отношения;\n"
        "- текущее место и время действия;\n"
        "- общий тон/стиль повествования (например: ироничный, мрачный, бытовой);\n"
        "- незакрытые сюжетные нити (если явно есть).\n\n"
        "Не пересказывай сюжет подробно. Не добавляй ничего, чего нет в тексте.\n"
        "Пиши по-русски, тезисно, короткими строками.\n"
        "Выведи ТОЛЬКО обновлённую памятку — без пояснений, заголовков и маркеров."
    )
    user = (
        "/no_think\n"
        f"Памятка должна быть не больше ~{max_chars} символов.\n\n"
        "ТЕКУЩАЯ ПАМЯТКА:\n---\n"
        f"{prev_summary or '(пока пусто)'}\n---\n\n"
        "НОВЫЙ ФРАГМЕНТ ПЕРЕВОДА:\n---\n"
        f"{joined}\n---"
    )

    raw = call_ollama(system, user, model=model, temperature=0.2,
                      num_predict=700, num_ctx=TRANSLATE_NUM_CTX)
    cleaned = _strip_model_noise(raw)
    if not cleaned or len(cleaned) < 10:
        return prev_summary
    if len(cleaned) > int(max_chars * 1.6):
        cleaned = cleaned[:int(max_chars * 1.6)].rstrip()
    return cleaned


# ============================================================
# ПЕРЕВОД
# ============================================================

def clean_translation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[\w]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

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

    korean_chars = sum(1 for c in translation if 0xAC00 <= ord(c) <= 0xD7AF)
    cyrillic_chars = sum(1 for c in translation if 0x0400 <= ord(c) <= 0x04FF)

    if korean_chars > len(translation) * 0.20:
        return False, f"много корейских символов ({korean_chars}/{len(translation)})"
    if cyrillic_chars < len(translation) * 0.30:
        return False, f"мало кириллицы ({cyrillic_chars}/{len(translation)})"

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
    running_summary: str = "",
) -> str:
    relevant = find_relevant_terms(glossary_terms, segment_text, max_terms=max_glossary)
    glossary_section = format_glossary_for_prompt(relevant)

    summary_section = ""
    if running_summary:
        summary_section = (
            "\nПАМЯТКА ПО ИСТОРИИ (для связности — НЕ переводить, НЕ повторять, "
            "НЕ добавлять в результат):\n"
            f"---\n{running_summary[:SUMMARY_MAX_CHARS]}\n---\n"
        )

    # Сырой контекст предыдущего сегмента — по умолчанию ВЫКЛ (источник галлюцинаций).
    context_section = ""
    if TRANSLATE_USE_PREV_CONTEXT and context_before:
        ctx = context_before[-TRANSLATE_PREV_CONTEXT_CHARS:]
        context_section = (
            "\nПРЕДЫДУЩИЙ ПЕРЕВОД (только для связности стиля — НЕ переводить и НЕ повторять):\n"
            f"---\n{ctx}\n---\n"
        )

    system = (
        "Ты — профессиональный литературный переводчик с корейского на русский, "
        "специализируешься на корейских ранобэ и веб-новеллах.\n"
        "Переведи приведённый корейский текст на естественный, живой русский язык — "
        "так, как написал бы хороший русскоязычный автор.\n\n"
        "ПРАВИЛА:\n"
        "1. Точный смысл, но русский синтаксис и ритм. Не калькируй корейский порядок слов.\n"
        "2. Сохраняй стиль и тон оригинала: сленг → русским сленгом, формальное → формально, грубое → грубо.\n"
        "3. ОБЯЗАТЕЛЬНО используй переводы из глоссария (он в сообщении) — имена/термины должны быть единообразными.\n"
        "   Если у термина в скобках указана ханча (漢字) — переводи по СМЫСЛУ иероглифов, как дано в глоссарии, а не фонетикой.\n"
        "4. Имена склоняй естественно по русским правилам:\n"
        "   - Юдер → Юдера, Юдеру, с Юдером\n"
        "   - Корделия → Корделии, Корделию\n"
        "   - Хон Юхи, Чхвирён — несклоняемые корейские (как «Окубо»)\n"
        "5. Сохраняй разбиение на абзацы и диалоги (каждая реплика с новой строки).\n"
        "6. Никаких пояснений, комментариев, заголовков, рассуждений — ТОЛЬКО перевод.\n"
        "7. Переводи ТОЛЬКО текст этого сегмента. НЕ добавляй реплик, действий или предложений, "
        "которых нет в корейском оригинале сегмента, даже если так «логичнее». Числа и имена переноси точно.\n"
        "8. Корейский игровой/сетевой сленг переводи русскими аналогами:\n"
        "   - 고인물 / 썩은물 → «олд», «прошарик», «задрот» (по контексту)\n"
        "   - 닥쳐 → «заткнись»\n"
        "   - 야 → «эй», «слушай»\n"
        "   - 헐 / 헉 → «офигеть», «ого»\n"
        "   - ㅋㅋㅋ → «ахах», «лол» (по контексту)"
    )

    user = (
        "/no_think\n"
        f"{summary_section}"
        "ГЛОССАРИЙ (используй ровно эти переводы; склоняй по русским правилам):\n"
        f"{glossary_section}\n"
        f"{context_section}\n"
        f"КОРЕЙСКИЙ ТЕКСТ (Сегмент #{segment_id}):\n"
        "---\n"
        f"{segment_text}\n"
        "---\n\n"
        "ПЕРЕВОД (только русский текст, без префиксов и комментариев):"
    )

    return call_ollama(system, user, model=model, temperature=temperature,
                       num_predict=4096, num_ctx=TRANSLATE_NUM_CTX)


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def atomic_save(output_data: dict, output_path: str):
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)


def build_output(translations: dict, total_segments: int,
                 model: str, glossary_count: int,
                 running_summary: str = "") -> dict:
    return {
        "metadata": {
            "model": model,
            "glossary_terms_count": glossary_count,
            "translated_segments": sum(1 for t in translations.values() if not t.get("error")),
            "total_segments": total_segments,
            "errors": sum(1 for t in translations.values() if t.get("error")),
            "running_summary": running_summary,
            "use_prev_context": TRANSLATE_USE_PREV_CONTEXT,
            "last_updated": datetime.now().isoformat(),
        },
        "translations": sorted(translations.values(), key=lambda x: x["id"]),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Этап 4: Перевод с глоссарием")
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
    parser.add_argument("--summary-every", type=int, default=SUMMARY_UPDATE_EVERY,
                        help="Обновлять бегущее саммари каждые N сегментов (0 = выкл)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Полностью отключить бегущее саммари")
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
    running_summary = ""
    if (args.resume or args.retry_errors) and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("translations", []):
                existing[item["id"]] = item
            running_summary = data.get("metadata", {}).get("running_summary", "") or ""
        print(f"Загружено {len(existing)} существующих переводов")
        if running_summary:
            print(f"Восстановлена памятка по истории ({len(running_summary)} симв)")

        if not args.no_backup:
            backup = os.path.join(BACKUP_DIR, os.path.basename(args.output)
                                   + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(args.output, backup)
            print(f"Бэкап: {backup}")

    summary_every = 0 if args.no_summary else max(0, args.summary_every)
    summary_enabled = summary_every > 0

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
    print(f"Бегущее саммари: {'каждые ' + str(summary_every) if summary_enabled else 'выкл'}")
    print(f"Сырой контекст пред. сегмента: {'ВКЛ' if TRANSLATE_USE_PREV_CONTEXT else 'выкл'}")
    print("-" * 60)

    translations = dict(existing)
    recent_finals = []
    run_start = time.time()
    processed_ok = 0
    processed_fail = 0

    for idx, seg_id in enumerate(to_translate_ids):
        segment = segments_by_id[seg_id]
        seg_text = segment["text"]
        char_count = segment.get("char_count", len(seg_text))

        # Контекст из предыдущего сегмента — только если включён в config
        context_before = ""
        if TRANSLATE_USE_PREV_CONTEXT:
            prev_id = seg_id - 1
            if prev_id in translations and not translations[prev_id].get("error"):
                context_before = translations[prev_id].get("translation", "")

        if processed_ok + processed_fail > 0:
            elapsed = time.time() - run_start
            avg = elapsed / (processed_ok + processed_fail)
            remaining = (len(to_translate_ids) - idx) * avg
            eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M %d.%m")
            eta_str = f" | ETA {eta} ({remaining/3600:.1f}ч)"
        else:
            eta_str = ""

        relevant_count = len(find_relevant_terms(glossary_terms, seg_text,
                                                 max_terms=args.max_glossary))

        print(f"[{idx+1}/{len(to_translate_ids)}] #{seg_id} "
              f"({char_count} симв | глос: {relevant_count}){eta_str}",
              flush=True)

        start_time = time.time()
        raw = translate_segment(
            seg_text, glossary_terms, context_before, seg_id,
            args.model, args.temperature, args.max_glossary,
            running_summary=running_summary,
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

            if summary_enabled:
                recent_finals.append(translation)
                if len(recent_finals) > SUMMARY_RECENT_SEGMENTS:
                    recent_finals = recent_finals[-SUMMARY_RECENT_SEGMENTS:]
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

        if (summary_enabled and ok and processed_ok > 0
                and processed_ok % summary_every == 0 and recent_finals):
            print("  · обновляю памятку по истории...", flush=True)
            running_summary = update_running_summary(
                running_summary, recent_finals, args.model
            )

        try:
            atomic_save(
                build_output(translations, len(segments), args.model,
                             len(glossary_terms), running_summary),
                args.output,
            )
        except Exception as e:
            print(f"  !! Ошибка сохранения: {e}")

        if idx < len(to_translate_ids) - 1:
            time.sleep(args.delay)

    total_time = time.time() - run_start
    print("\n" + "=" * 60)
    print(f"Готово за {total_time/60:.1f} минут")
    print(f"Успешно: {processed_ok}, Ошибок: {processed_fail}")
    print(f"Файл: {args.output}")
    if processed_fail > 0:
        print("\nДля повторной попытки ошибочных:")
        print(f"  python {os.path.basename(sys.argv[0])} --retry-errors")
    print("Дальше: python stage5_translate_check.py")


if __name__ == "__main__":
    main()
