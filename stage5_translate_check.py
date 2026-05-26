"""
Этап 5: Кросс-проверка ПЕРЕВОДА независимой моделью (по умолчанию aya-expanse).

Для КАЖДОГО сегмента модель сверяет русский перевод (от qwen) с корейским
оригиналом и исправляет ТОЛЬКО смысл. Стиль не трогает — красотой займётся
stage6 (gemma).

Что нового против ver1.0:
  - ВЫРАВНИВАНИЕ РЕПЛИК: модель обязана сверить число реплик/предложений с
    корейским и УДАЛИТЬ строки, которых в оригинале нет (это ловит галлюцинации-
    копии вроде дублированного диалога в #25), а не только править лексику.
  - valid() переякорен: корректное СОКРАЩЕНИЕ (удаление галлюцинации) больше не
    откатывается. Ограничен только РОСТ относительно черновика (чтобы сам
    проверяльщик не дописал отсебятину) и грубое укорочение относительно
    корейского источника (реальный пропуск).
  - Глоссарий в подсказке показывает hanja.

Полностью автоматически. «Сырой» перевод qwen сохраняется в 3_translated.qwen.json,
3_translated.json перезаписывается исправленной версией (поле translation).
Поддерживает --resume.

Запуск:  python stage5_translate_check.py
         python stage5_translate_check.py --resume
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


MAX_GLOSSARY_TERMS = 60

CHECK_SYSTEM = (
    "Ты — двуязычный контролёр точности перевода с корейского на русский.\n"
    "Тебе дают корейский оригинал и его русский перевод. Сверь СМЫСЛ и СТРУКТУРУ и "
    "исправь ТОЛЬКО ошибки:\n"
    "- неверно понятые слова и омонимы (включая ханча-омонимы);\n"
    "- перепутанные числа;\n"
    "- заимствования (корейская транслитерация иностранных слов: Fly, Outboxer, "
    "Frost Anvil, Kaplan и т.п.);\n"
    "- имена и термины — строго по глоссарию (он будет в сообщении); если у термина "
    "указана ханча — перевод по смыслу иероглифов, не фонетикой.\n\n"
    "ВЫРАВНИВАНИЕ (важнее всего):\n"
    "- Перевод должен соответствовать оригиналу: НИЧЕГО ЛИШНЕГО и ничего потерянного.\n"
    "- Сверь число реплик и предложений с корейским. Если в русском есть реплика, "
    "ответ, реакция или предложение, которых НЕТ в корейском оригинале сегмента — "
    "УДАЛИ их полностью. Это галлюцинация (часто — копия из соседней сцены).\n"
    "- Если в оригинале N реплик прямой речи, в переводе должно быть ровно N.\n\n"
    "ПРИМЕР НАРУШЕНИЯ (исправь так же):\n"
    "  Корейский: \"도련님이 오셨습니다.\"   (одна реплика — служанка объявляет)\n"
    "  Неверный перевод:\n"
    "    — Молодой господин пришёл.\n"
    "    — А? Да, всё в порядке. Сейчас пойду.\n"
    "  Правильно (вторую реплику удалить — её нет в корейском):\n"
    "    — Молодой господин пришёл.\n\n"
    "НЕ улучшай стиль, НЕ переписывай удачные места, НЕ меняй разбивку на абзацы и "
    "реплики (кроме удаления лишних). Держись как можно ближе к исходным формулировкам "
    "— меняй ТОЛЬКО то, что искажает смысл или отсутствует в оригинале. Если всё верно "
    "— верни перевод без изменений.\n"
    "Выводи ТОЛЬКО исправленный русский текст, без пояснений, заголовков и маркеров."
)


def call_ollama(system, user, model, temperature, num_predict=4096,
                num_ctx=CHECK_NUM_CTX, keep_alive="30m", max_retries=3):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    last = ""
    for a in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=1200)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            print("\nОШИБКА: Ollama недоступна. Запустите: ollama serve")
            sys.exit(1)
        except requests.exceptions.Timeout:
            last = "timeout"
            print(f"\n  ! Timeout ({a}/{max_retries})...")
            time.sleep(15)
        except requests.exceptions.HTTPError as e:
            print(f"\nОШИБКА HTTP: {e}\n  Ответ Ollama: {getattr(e.response, 'text', '')}")
            print("  Подсказка: 404 = модель не найдена. Сверьте MODEL_CHECK с `ollama list`.")
            return ""
        except Exception as e:
            last = str(e)
            print(f"\n  ! Ошибка: {e} ({a}/{max_retries})")
            time.sleep(5)
    print(f"  ! Не удалось за {max_retries} попыток ({last})")
    return ""


def relevant_glossary(terms, source, max_terms=MAX_GLOSSARY_TERMS):
    m = []
    for t in terms:
        k = (t.get("korean") or "").strip()
        if k and k in source:
            m.append((source.count(k), len(k), t))
    m.sort(key=lambda x: (-x[0], -x[1]))
    out = m[:max_terms]
    if not out:
        return ""
    lines = ["ГЛОССАРИЙ (имена/термины — строго так):"]
    for _, _, t in out:
        hanja = (t.get("hanja") or "").strip()
        head = f"{t.get('korean')} ({hanja})" if hanja else f"{t.get('korean')}"
        lines.append(f"  {head} → {t.get('russian')}")
    return "\n".join(lines) + "\n\n"


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[\w]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"^(Исправленный перевод|Исправленный русский перевод|Перевод|Результат"
                  r"|Corrected translation)[:：]?\s*\n?", "", text, flags=re.IGNORECASE)
    return text.strip()


def valid(corrected, draft, source):
    """Валидация исправления.
    Разрешаем СОКРАЩЕНИЕ (удаление галлюцинаций), ограничиваем РОСТ относительно
    черновика (чтобы проверяльщик сам не дописал) и грубое укорочение
    относительно корейского источника (реальный пропуск)."""
    if not corrected or len(corrected) < 10:
        return False, "пусто/слишком коротко"
    cyr = sum(1 for c in corrected if 0x0400 <= ord(c) <= 0x04FF)
    if cyr < len(corrected) * 0.30:
        return False, "мало кириллицы"
    kor = sum(1 for c in corrected if 0xAC00 <= ord(c) <= 0xD7AF)
    if kor > len(corrected) * 0.15:
        return False, "много корейского"
    # Грубый пропуск: перевод не может быть сильно короче корейского источника
    if len(corrected) < len(source) * 0.40:
        return False, "слишком коротко относительно оригинала (возможен пропуск)"
    # Проверяльщик дописал отсебятину: рост относительно черновика ограничен
    if len(corrected) > len(draft) * 1.6:
        return False, "сильно длиннее черновика (возможна дописка)"
    # Укорочение относительно черновика — РАЗРЕШЕНО (это удаление галлюцинаций)
    return True, ""


def check_segment(source, translation, glossary_terms, model, temperature):
    gloss = relevant_glossary(glossary_terms, source)
    user = (f"{gloss}КОРЕЙСКИЙ ОРИГИНАЛ:\n---\n{source}\n---\n\n"
            f"РУССКИЙ ПЕРЕВОД:\n---\n{translation}\n---\n\n"
            "Исправленный русский перевод:")
    raw = call_ollama(CHECK_SYSTEM, user, model=model, temperature=temperature)
    return clean(raw) if raw else ""


def atomic_save(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description="Этап 5: кросс-проверка перевода (aya)")
    p.add_argument("--input", default=TRANSLATED_FILE)
    p.add_argument("--glossary", default=GLOSSARY_FILE)
    p.add_argument("--model", default=MODEL_CHECK)
    p.add_argument("--temperature", type=float, default=CHECK_TEMPERATURE)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--delay", type=float, default=0.5)
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ОШИБКА: нет файла {args.input}")
        sys.exit(1)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["translations"]
    by_id = {t["id"]: t for t in items}

    glossary = []
    if os.path.exists(args.glossary):
        with open(args.glossary, "r", encoding="utf-8") as f:
            glossary = json.load(f).get("terms", [])

    raw_backup = os.path.join(OUTPUT_DIR, "3_translated.qwen.json")
    if not os.path.exists(raw_backup):
        shutil.copy2(args.input, raw_backup)
        print(f"Бэкап qwen-перевода: {raw_backup}")

    end = args.end if args.end > 0 else max(t["id"] for t in items)
    todo = [t["id"] for t in items if args.start <= t["id"] <= end]
    if args.resume:
        todo = [i for i in todo if not by_id[i].get("checked")]
    todo = [i for i in todo if not by_id[i].get("error")]  # битые переводы пропускаем

    if not todo:
        print("Нечего проверять.")
        return

    print(f"Проверка перевода: {len(todo)} сегментов | модель {args.model}")
    print("-" * 60)

    start = time.time()
    done = 0
    changed_total = 0
    for idx, sid in enumerate(todo):
        it = by_id[sid]
        src = it["source"]
        tr = it.get("translation", "")
        if done > 0:
            avg = (time.time() - start) / done
            eta = (datetime.now() + timedelta(seconds=(len(todo) - idx) * avg)).strftime("%H:%M %d.%m")
            es = f" | ETA {eta}"
        else:
            es = ""
        print(f"[{idx+1}/{len(todo)}] #{sid} ({len(src)} симв){es}", flush=True)
        t0 = time.time()
        corrected = check_segment(src, tr, glossary, args.model, args.temperature)
        ok, why = valid(corrected, tr, src)
        if ok:
            changed = corrected.strip() != tr.strip()
            it.setdefault("qwen_translation", tr)
            it["translation"] = corrected
            it["checked"] = True
            it["check_changed"] = changed
            it["check_model"] = args.model
            if changed:
                changed_total += 1
            print(f"  {'ИСПРАВЛЕНО' if changed else 'ок (без изменений)'} ({time.time()-t0:.0f}с)")
        else:
            it["checked"] = True
            it["check_changed"] = False
            it["check_failed"] = True
            it["check_fail_reason"] = why
            print(f"  оставлено как есть (проверка не прошла: {why}, {time.time()-t0:.0f}с)")
        done += 1
        atomic_save({
            "metadata": {
                "check_model": args.model,
                "checked": sum(1 for t in items if t.get("checked")),
                "changed": sum(1 for t in items if t.get("check_changed")),
                "total": len(items),
                "last_updated": datetime.now().isoformat(),
            },
            "translations": sorted(items, key=lambda x: x["id"]),
        }, args.input)
        if idx < len(todo) - 1:
            time.sleep(args.delay)

    print("\n" + "=" * 60)
    print(f"Готово. Исправлено сегментов: {changed_total}/{len(todo)}")
    print(f"Файл обновлён: {args.input} (сырой qwen — в {raw_backup})")
    print("Дальше: python stage6_proofread.py --export-txt")


if __name__ == "__main__":
    main()
