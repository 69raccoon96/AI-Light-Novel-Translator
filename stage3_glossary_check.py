"""
Этап 3: Кросс-проверка глоссария НЕЗАВИСИМОЙ моделью (по умолчанию aya-expanse).

ГАРАНТИИ:
  - korean / romanization / hanja НЕПРИКОСНОВЕННЫ: проверяющая модель меняет
    только russian/category/note. Ключи всегда берутся из оригинала qwen
    (иначе модель портит ключ, напр. 코델리아→«코델ия», и термин перестаёт
    находиться в тексте — тихий пропуск).
  - База берётся из чистого qwen (2_glossary.qwen.json) → повторный запуск
    идемпотентен.
  - BUILTIN-гоноративы (посеянные stage2) НЕ отправляются на проверку и
    переносятся как есть — это фиксированный детерминированный слой.
  - Ручные оверрайды (2_glossary_overrides.json) — высший приоритет.
  - Разногласия (где проверяльщик изменил перевод qwen) → 2_glossary_disagreements.txt.

Главный фокус: заимствования (ㅍ=F, ㄹ=L), ханча-омонимы, числа, титулы, и
ОСОБЕННО — концепт-термины, которые qwen мог дать фонетикой вместо смысла
(«Гуин Цзюймэй» вместо «меридианы Девяти Инь»): если есть hanja — перевод
строится по смыслу иероглифов.

Запуск:  python stage3_glossary_check.py
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime

import requests

from config import *


CHECK_SYSTEM = """You are an independent reviewer of a Korean→Russian glossary produced by another model.
Find and FIX mistakes in the RUSSIAN side only:
- CONCEPT TERMS via HANJA: if an item has a "hanja" field, translate the Russian by the
  MEANING of those characters, NOT by the Korean sound. A bare phonetic transliteration
  of a concept term is an ERROR you must fix:
    九陰絶脈 → «меридианы Девяти Инь»   (NOT «Гуин Цзюймэй»)
    天武之體 → «тело Небесного Воина»   (NOT «Чхонму Чхи»)
- FOREIGN LOANWORDS: recover the original foreign word, do not transliterate blindly
  (플라이→Fly, 아웃복서→Outboxer/Аутбоксер, 프로스트 앤빌→Frost Anvil, 카플란→Kaplan; ㅍ=F not П, ㄹ=L not Р).
- HANJA HOMOPHONES: choose the right meaning by context (변경 邊境 граница→«маркграф», NOT 變更 change; 성 聖 holy).
- NUMBERS: 구=9, 십=10, 삼=3 — never swap.
- KOREAN NOBILITY TITLES (fixed mapping): 공작=герцог, 대공=великий герцог, 후작=маркиз,
  변경백=маркграф, 백작=граф, 자작=виконт, 남작=барон. 백작가 = «графский род / дом графа» (НЕ барон!).
- Wrong Kontsevich romanization or wrong gender of names.
Do NOT change the korean or hanja fields. Do NOT invent new terms. If a term is already correct, keep it.
Respond with STRICT JSON ONLY, no commentary:
{"terms":[{"korean":"...","russian":"...","category":"...","note":"..."}]}"""


def call_ollama(system, user, model, num_ctx=CHECK_NUM_CTX,
                temperature=CHECK_TEMPERATURE, num_predict=4096):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    try:
        r = requests.post(url, json=payload, timeout=1200)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        print("ОШИБКА: Ollama недоступна. Запустите: ollama serve")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ОШИБКА HTTP: {e}\n  Ответ Ollama: {getattr(e.response, 'text', '')}")
        print("  Подсказка: 404 = модель не найдена. Сверьте MODEL_CHECK с `ollama list`.")
        return ""
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return ""


def parse_json(resp):
    resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
    m = re.search(r"\{[\s\S]*\}", resp)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    a = re.search(r"\[[\s\S]*\]", resp)
    if a:
        try:
            return {"terms": json.loads(a.group())}
        except json.JSONDecodeError:
            pass
    return {"terms": []}


def main():
    p = argparse.ArgumentParser(description="Этап 3: кросс-проверка глоссария")
    p.add_argument("--input", default=GLOSSARY_FILE)
    p.add_argument("--model", default=MODEL_CHECK)
    p.add_argument("--chunk", type=int, default=GLOSSARY_NORMALIZE_CHUNK)
    args = p.parse_args()

    raw_backup = os.path.join(OUTPUT_DIR, "2_glossary.qwen.json")

    # База = всегда чистый qwen. Если бэкапа ещё нет — создаём его из текущего input.
    if os.path.exists(raw_backup):
        base_path = raw_backup
    elif os.path.exists(args.input):
        shutil.copy2(args.input, raw_backup)
        base_path = raw_backup
        print(f"Бэкап qwen-глоссария: {raw_backup}")
    else:
        print(f"ОШИБКА: нет файла {args.input}")
        sys.exit(1)

    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    terms = base.get("terms", [])
    if not terms:
        print("Глоссарий пуст.")
        return

    # Оригиналы по корейскому ключу — источник истины для korean/romanization/hanja.
    originals = {}
    result = {}
    order = []
    for t in terms:
        k = (t.get("korean") or "").strip()
        if not k or k in originals:
            continue
        originals[k] = t
        result[k] = dict(t)        # стартуем от qwen, поверх накладываем правки aya
        order.append(k)

    # BUILTIN-гоноративы на проверку не отправляем (детерминированный слой)
    review_keys = [k for k in order if not originals[k].get("builtin")]
    builtin_keys = [k for k in order if originals[k].get("builtin")]
    if builtin_keys:
        print(f"Builtin-гоноративы (без проверки): {len(builtin_keys)}")

    disagreements = []
    total = (len(review_keys) + args.chunk - 1) // args.chunk if review_keys else 0
    print(f"Проверка глоссария: {len(review_keys)} терминов, модель {args.model}")
    print(f"База (чистый qwen): {base_path}")
    print("-" * 60)

    for ci in range(0, len(review_keys), args.chunk):
        keys = review_keys[ci:ci + args.chunk]
        compact = [{"korean": k,
                    "hanja": originals[k].get("hanja", ""),
                    "russian": originals[k].get("russian"),
                    "category": originals[k].get("category"),
                    "note": (originals[k].get("note", "") or "")[:80]} for k in keys]
        n = ci // args.chunk + 1
        print(f"  Чанк {n}/{total} ({len(keys)})...", end=" ", flush=True)
        resp = call_ollama(CHECK_SYSTEM, json.dumps(compact, ensure_ascii=False, indent=2), args.model)
        got = parse_json(resp).get("terms", []) if resp else []
        applied = 0
        for t in got:
            if not isinstance(t, dict):
                continue
            k = (t.get("korean") or "").strip()
            if k not in originals:        # КЛЮЧ НЕПРИКОСНОВЕНЕН: чужие/искажённые игнорируем
                continue
            if originals[k].get("builtin"):  # builtin не трогаем даже если модель прислала
                continue
            new_ru = (t.get("russian") or "").strip()
            if new_ru and new_ru != (originals[k].get("russian") or "").strip():
                disagreements.append((k, originals[k].get("russian", ""), new_ru))
                result[k]["russian"] = new_ru
                applied += 1
            new_cat = (t.get("category") or "").strip()
            if new_cat:
                result[k]["category"] = new_cat
            new_note = (t.get("note") or "").strip()
            if new_note:
                result[k]["note"] = new_note
        print(f"правок: {applied}")
        time.sleep(0.5)

    fixed = [result[k] for k in order]

    # Ручные/облачные оверрайды — наивысший приоритет, переживают пере-прогон stage3.
    # Формат: {"terms":[{"korean":"..","russian":"..","note":".."}]}  ИЛИ  {"한국어":"перевод"}
    overrides_path = GLOSSARY_OVERRIDES_FILE
    if os.path.exists(overrides_path):
        try:
            with open(overrides_path, "r", encoding="utf-8") as f:
                ov_raw = json.load(f)
            ov = {}
            if isinstance(ov_raw, dict) and "terms" in ov_raw:
                for o in ov_raw["terms"]:
                    kk = (o.get("korean") or "").strip()
                    if kk:
                        ov[kk] = o
            elif isinstance(ov_raw, dict):
                ov = {k.strip(): {"russian": v} for k, v in ov_raw.items()}
            n_ov = 0
            for t in fixed:
                o = ov.get(t.get("korean"))
                if not o:
                    continue
                if o.get("russian"):
                    t["russian"] = o["russian"]
                if o.get("note"):
                    t["note"] = o["note"]
                n_ov += 1
            if n_ov:
                print(f"Применено ручных оверрайдов: {n_ov} (из {overrides_path})")
        except Exception as e:
            print(f"  ! Оверрайды не применены: {e}")

    fixed.sort(key=lambda x: (x.get("category", ""), x.get("korean", "")))

    out = {"metadata": {"check_model": args.model, "total_terms": len(fixed),
                        "disagreements": len(disagreements), "source": "cross-check"},
           "terms": fixed}
    tmp = args.input + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.input)

    # Отчёт разногласий — кандидаты на арбитраж Gemini (gemini_fallback --mode glossary)
    dis_path = os.path.join(OUTPUT_DIR, "2_glossary_disagreements.txt")
    with open(dis_path, "w", encoding="utf-8") as f:
        f.write(f"Разногласий: {len(disagreements)}  (qwen → {args.model})\n")
        f.write("=" * 60 + "\n")
        for k, q, a in disagreements:
            hj = originals[k].get("hanja", "")
            f.write(f"{k}{(' ['+hj+']') if hj else ''}\n  qwen: {q}\n  aya : {a}\n\n")

    # Структурированный JSON — для автоматического арбитража Gemini
    dis_json_path = os.path.join(OUTPUT_DIR, "2_glossary_disagreements.json")
    dis_json = {
        "metadata": {
            "check_model": args.model,
            "disagreements": len(disagreements),
            "generated_at": datetime.now().isoformat(),
        },
        "disagreements": [
            {
                "korean": k,
                "hanja": originals[k].get("hanja", ""),
                "category": originals[k].get("category", ""),
                "note": (originals[k].get("note", "") or ""),
                "qwen": q,
                "aya": a,
            }
            for k, q, a in disagreements
        ],
    }
    with open(dis_json_path, "w", encoding="utf-8") as f:
        json.dump(dis_json, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. {args.input} обновлён (korean/hanja-ключи сохранены от qwen).")
    print(f"Разногласий: {len(disagreements)} → {dis_path}")
    print(f"                            {dis_json_path}")
    print("Дальше: python gemini_fallback.py --mode glossary  (арбитр разногласий)")
    print("Или сразу: python stage4_translate.py")


if __name__ == "__main__":
    main()
