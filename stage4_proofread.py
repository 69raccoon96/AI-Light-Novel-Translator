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

MARK_START = "<<<IMPROVED>>>"
MARK_END = "<<<END>>>"

MAX_GLOSSARY_TERMS = 60


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(
    prompt: str,
    model: str = MODEL_PROOFREAD,
    temperature: float = PROOFREAD_TEMPERATURE,
    num_predict: int = 4096,
    num_ctx: int = 8192,
    keep_alive: str = "30m",
    max_retries: int = 3,
) -> str:
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
            print(f"\n  ! Timeout (попытка {attempt}/{max_retries})...")
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

def find_relevant_terms(glossary_terms: list, source_text: str,
                        max_terms: int = MAX_GLOSSARY_TERMS) -> list:
    """Только термины, реально встречающиеся в корейском оригинале сегмента."""
    matched = []
    for t in glossary_terms:
        kor = (t.get("korean") or "").strip()
        if kor and kor in source_text:
            matched.append((source_text.count(kor), len(kor), t))
    matched.sort(key=lambda x: (-x[0], -x[1]))
    return [m[2] for m in matched[:max_terms]]


def format_glossary_for_prompt(relevant_terms: list) -> str:
    if not relevant_terms:
        return ""
    lines = ["ГЛОССАРИЙ (используй ровно эти переводы для имён/терминов):"]
    for t in relevant_terms:
        lines.append(f"  {t.get('korean')} → {t.get('russian')}")
    return "\n".join(lines) + "\n\n"


# ============================================================
# ВЫЧИТКА
# ============================================================

def proofread_segment(
    source: str,
    translation: str,
    context_before: str = "",
    glossary_terms: list = None,
    mode: str = "full",
    model: str = MODEL_PROOFREAD,
    temperature: float = PROOFREAD_TEMPERATURE,
) -> dict:
    """ВАЖНО: context_after намеренно не используется — он провоцировал галлюцинации."""

    glossary_section = ""
    if glossary_terms:
        relevant = find_relevant_terms(glossary_terms, source)
        glossary_section = format_glossary_for_prompt(relevant)

    context_section = ""
    if context_before:
        ctx = context_before[-500:]
        context_section = (
            "ПРЕДЫДУЩИЙ ПЕРЕВОД (только для согласования стиля — НЕ копировать в результат):\n"
            f"---\n{ctx}\n---\n\n"
        )

    if mode == "full":
        prompt = f"""Ты — редактор русскоязычного перевода с корейского. Твой текст потом будут читать как книгу.

ЗАДАЧА: улучшить ТОЛЬКО приведённый русский перевод. Сверься с корейским оригиналом, чтобы убедиться в точности смысла.

СТРОГИЕ ПРАВИЛА:
1. НЕ добавляй новых предложений, реплик, описаний, которых нет в корейском оригинале.
2. НЕ копируй текст из «предыдущего перевода» — он только для согласования стиля.
3. Длина результата должна быть близка к длине входного перевода (±20%).
4. Имена и термины — СТРОГО по глоссарию, не меняй их транслитерацию.
5. Правь грамматику, пунктуацию, неестественные обороты, кальки с корейского.
6. Сохраняй абзацы, реплики, переносы строк ровно как в исходном переводе.
7. Если перевод уже хорош — верни его почти без изменений.
8. Стилистика — живая русская проза, а не подстрочник.

{glossary_section}{context_section}КОРЕЙСКИЙ ОРИГИНАЛ:
---
{source}
---

ТЕКУЩИЙ РУССКИЙ ПЕРЕВОД:
---
{translation}
---

Выведи ТОЛЬКО улучшенный текст между маркерами. Никаких комментариев,
объяснений, JSON. Только маркеры и текст между ними.

{MARK_START}
(улучшенный перевод здесь)
{MARK_END}
"""
    else:  # light
        prompt = f"""Ты — корректор русского текста.

Сделай МИНИМАЛЬНУЮ правку: только явные ошибки грамматики, пунктуации и неуклюжие фразы.
НЕ переписывай. НЕ добавляй ничего нового. НЕ меняй имена.
Длина должна остаться примерно той же.

{glossary_section}РУССКИЙ ТЕКСТ:
---
{translation}
---

Выведи ТОЛЬКО результат между маркерами:

{MARK_START}
(текст здесь)
{MARK_END}
"""

    response = call_ollama(prompt, model=model, temperature=temperature)
    return parse_proofread_response(response, translation)


def parse_proofread_response(response: str, original: str) -> dict:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Gemma3 иногда вместо маркеров пишет ```text ... ```
    response = re.sub(r"```\w*\n", "", response)
    response = re.sub(r"```", "", response)

    pattern = re.escape(MARK_START) + r"(.*?)" + re.escape(MARK_END)
    m = re.search(pattern, response, re.DOTALL)

    if m:
        improved = m.group(1).strip()
    else:
        m2 = re.search(re.escape(MARK_START) + r"(.*)", response, re.DOTALL)
        if m2:
            improved = m2.group(1).strip()
        else:
            return {"improved_text": original, "fallback_reason": "no_markers"}

    improved = improved.replace(MARK_START, "").replace(MARK_END, "").strip()

    # Чистим Gemma3-преамбулы, которые могли просочиться внутрь маркеров
    gemma_prefixes = [
        r"^(Конечно|Понял|Хорошо)[,!\s]*\s*(вот|вот\s+\w+|вот.*?перевод)?[:：]?\s*\n?",
        r"^(Here'?s|Here is)\s+\w+[:：]?\s*\n?",
        r"^\*\*[^*]+\*\*[:：]?\s*\n",
        r"^Улучшенн[а-я]+\s+(текст|версия|перевод)[:：]?\s*\n?",
    ]
    for p in gemma_prefixes:
        improved = re.sub(p, "", improved, flags=re.IGNORECASE)
    improved = improved.strip()

    # Убираем случайные **bold** и *italic* от Gemma3
    improved = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", improved)
    improved = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", improved)

    # === SANITY CHECKS ===
    if not improved or len(improved) < 10:
        return {"improved_text": original, "fallback_reason": "too_short_abs"}

    ratio = len(improved) / max(len(original), 1)
    #if ratio > 1.6:
    #    print(improved)
    #    return {"improved_text": original,
    #            "fallback_reason": f"too_long_ratio={ratio:.2f}"}
    #if ratio < 0.4:
    #    return {"improved_text": original,
    #            "fallback_reason": f"too_short_ratio={ratio:.2f}"}

    # Защита от утечки промпта
    leak_markers = ["КОРЕЙСКИЙ ОРИГИНАЛ", "ГЛОССАРИЙ", "<think>",
                    "ПРЕДЫДУЩИЙ ПЕРЕВОД", "ТЕКУЩИЙ РУССКИЙ",
                    MARK_START, MARK_END]
    for lm in leak_markers:
        if lm in improved:
            return {"improved_text": original, "fallback_reason": f"leak:{lm}"}

    # Не вернулась ли куча корейского
    korean_chars = sum(1 for c in improved if "\uac00" <= c <= "\ud7af")
    if korean_chars > len(improved) * 0.15:
        return {"improved_text": original,
                "fallback_reason": f"too_much_korean={korean_chars}"}

    return {"improved_text": improved}


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def atomic_save(output_data: dict, output_path: str):
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)


def make_result_item(seg_id, source, draft, final, fallback=None,
                     skipped=False, skip_reason=None, elapsed=None):
    """Унифицированная структура результата для ВСЕХ случаев."""
    item = {
        "id": seg_id,
        "source": source,
        "draft_translation": draft,
        "final_translation": final,
    }
    if fallback:
        item["fallback_reason"] = fallback
    if skipped:
        item["skipped"] = True
        item["skip_reason"] = skip_reason
    if elapsed is not None:
        item["time_seconds"] = round(elapsed, 1)
    return item


def build_output(results: list, model: str, mode: str) -> dict:
    return {
        "metadata": {
            "model": model,
            "mode": mode,
            "proofread_segments": len(results),
            "fallbacks": sum(1 for r in results if r.get("fallback_reason")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "last_updated": datetime.now().isoformat(),
        },
        "results": sorted(results, key=lambda x: x["id"]),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Этап 4: Вычитка")
    parser.add_argument("--input", type=str, default=TRANSLATED_FILE)
    parser.add_argument("--glossary", type=str, default=GLOSSARY_FILE)
    parser.add_argument("--output", type=str, default=FINAL_FILE)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--mode", choices=["full", "light"], default="full")
    parser.add_argument("--model", type=str, default=MODEL_PROOFREAD)
    parser.add_argument("--temperature", type=float, default=PROOFREAD_TEMPERATURE)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-fallbacks", action="store_true",
                        help="Перезапустить только те, что упали в fallback")
    parser.add_argument("--export-txt", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ОШИБКА: Файл не найден: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        translations_data = json.load(f)
    translations = translations_data["translations"]
    trans_dict = {t["id"]: t for t in translations}

    glossary_terms = []
    if os.path.exists(args.glossary):
        with open(args.glossary, "r", encoding="utf-8") as f:
            glossary_terms = json.load(f).get("terms", [])
        print(f"Глоссарий: {len(glossary_terms)} терминов")

    end_id = args.end if args.end > 0 else max(t["id"] for t in translations)

    # --- Загрузка существующих ---
    existing = {}
    if (args.resume or args.retry_fallbacks) and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for item in json.load(f).get("results", []):
                existing[item["id"]] = item
        print(f"Загружено {len(existing)} уже обработанных")

        if not args.no_backup:
            backup = args.output + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(args.output, backup)
            print(f"Бэкап: {backup}")

    # --- Что обрабатывать ---
    if args.retry_fallbacks:
        to_proofread_ids = [
            sid for sid, r in existing.items()
            if r.get("fallback_reason") and args.start <= sid <= end_id
        ]
        to_proofread_ids.sort()
        print(f"Режим RETRY-FALLBACKS: {len(to_proofread_ids)} сегментов")
    elif args.resume:
        to_proofread_ids = [
            t["id"] for t in translations
            if args.start <= t["id"] <= end_id and t["id"] not in existing
        ]
    else:
        to_proofread_ids = [
            t["id"] for t in translations
            if args.start <= t["id"] <= end_id
        ]

    if not to_proofread_ids:
        print("Нечего вычитывать.")
        return

    print(f"\nК вычитке: {len(to_proofread_ids)} сегментов")
    print(f"Модель: {args.model} | режим: {args.mode} | темп: {args.temperature}")
    print("-" * 60)

    results = dict(existing)
    run_start = time.time()
    processed = 0
    fallback_count = 0

    for idx, seg_id in enumerate(to_proofread_ids):
        item = trans_dict.get(seg_id)
        if not item:
            print(f"[{idx+1}/{len(to_proofread_ids)}] #{seg_id} — нет в input, пропуск")
            continue

        # Пропускаем сегменты с ошибкой перевода
        if item.get("error"):
            results[seg_id] = make_result_item(
                seg_id=seg_id,
                source=item["source"],
                draft=item.get("translation", ""),
                final=item.get("translation", ""),  # оставляем как есть
                skipped=True,
                skip_reason="translation_error",
            )
            atomic_save(build_output(list(results.values()), args.model, args.mode), args.output)
            print(f"[{idx+1}/{len(to_proofread_ids)}] #{seg_id} SKIP (ошибка перевода)")
            continue

        # Контекст из ПРЕДЫДУЩЕГО уже обработанного (final)
        prev_id = seg_id - 1
        context_before = ""
        if prev_id in results and not results[prev_id].get("skipped"):
            context_before = results[prev_id].get("final_translation", "")
        elif prev_id in trans_dict:
            # На первом проходе используем черновой
            context_before = trans_dict[prev_id].get("translation", "")

        # ETA
        if processed > 0:
            elapsed = time.time() - run_start
            avg = elapsed / processed
            remaining = (len(to_proofread_ids) - idx) * avg
            eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M %d.%m")
            eta_str = f" | ETA {eta} ({remaining/3600:.1f}ч)"
        else:
            eta_str = ""

        print(f"[{idx+1}/{len(to_proofread_ids)}] #{seg_id} "
              f"({len(item['source'])} симв){eta_str}", flush=True)

        start_time = time.time()
        result = proofread_segment(
            source=item["source"],
            translation=item["translation"],
            context_before=context_before,
            glossary_terms=glossary_terms,
            mode=args.mode,
            model=args.model,
            temperature=args.temperature,
        )
        elapsed = time.time() - start_time

        improved = result.get("improved_text", item["translation"])
        fallback = result.get("fallback_reason")

        results[seg_id] = make_result_item(
            seg_id=seg_id,
            source=item["source"],
            draft=item["translation"],
            final=improved,
            fallback=fallback,
            elapsed=elapsed,
        )

        processed += 1
        if fallback:
            fallback_count += 1
            print(f"  FALLBACK ({fallback}, {elapsed:.0f}с)")
        else:
            preview = improved[:90].replace("\n", " ")
            if len(improved) > 90:
                preview += "..."
            print(f"  OK ({elapsed:.0f}с, {len(improved)} симв)")
            print(f"  → {preview}")

        # Атомарное сохранение после КАЖДОГО сегмента
        try:
            atomic_save(build_output(list(results.values()), args.model, args.mode),
                        args.output)
        except Exception as e:
            print(f"  !! Ошибка сохранения: {e}")

        if idx < len(to_proofread_ids) - 1:
            time.sleep(args.delay)

    # --- Итоги ---
    total_time = time.time() - run_start
    print("\n" + "=" * 60)
    print(f"Готово за {total_time/60:.1f} минут")
    print(f"Обработано: {processed}, fallback: {fallback_count}, "
          f"всего в файле: {len(results)}")
    print(f"Файл: {args.output}")

    if fallback_count > 0:
        print(f"\nДля повторной попытки fallback'ов:")
        print(f"  python {os.path.basename(sys.argv[0])} --retry-fallbacks")

    # --- Экспорт txt ---
    if args.export_txt:
        txt_path = args.output.replace(".json", ".txt")
        sorted_results = sorted(results.values(), key=lambda x: x["id"])
        with open(txt_path, "w", encoding="utf-8") as f:
            for item in sorted_results:
                text = item.get("final_translation") or item.get("draft_translation", "")
                if text:
                    f.write(text)
                    f.write("\n\n")
        print(f"TXT: {txt_path}")


if __name__ == "__main__":
    main()
