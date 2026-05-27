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
  - Разногласия (где проверяльщик изменил перевод qwen) → 2_glossary_disagreements.txt
    + JSON для арбитра Gemini.

RESUME (важно):
  - Прогресс сохраняется ПОСЛЕ КАЖДОГО ЧАНКА в 2_glossary.aya.checkpoint.json.
  - При запуске с --resume и наличии чекпойнта продолжаем с того места, на
    котором остановились.
  - Маркер завершения — 2_glossary.checked.json (пишется только в самом конце).
    Если он есть и запуск с --resume — выходим сразу без работы.
  - Чекпойнт удаляется по завершении стадии.

Запуск:
  python stage3_glossary_check.py               # с нуля от qwen
  python stage3_glossary_check.py --resume      # продолжить с чекпойнта (или с нуля если его нет)
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


# --- пути ---
QWEN_BACKUP = os.path.join(OUTPUT_DIR, "2_glossary.qwen.json")
CHECKED_MARKER = os.path.join(OUTPUT_DIR, "2_glossary.checked.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "2_glossary.aya.checkpoint.json")
DISAGREE_TXT = os.path.join(OUTPUT_DIR, "2_glossary_disagreements.txt")
DISAGREE_JSON = os.path.join(OUTPUT_DIR, "2_glossary_disagreements.json")


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


def atomic_save(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def save_checkpoint(result, disagreements, order, review_keys,
                    chunks_done, total_chunks, model):
    """Сохраняет текущее состояние работы. Атомарно."""
    data = {
        "metadata": {
            "check_model": model,
            "chunks_done": chunks_done,
            "total_chunks": total_chunks,
            "review_keys_total": len(review_keys),
            "last_updated": datetime.now().isoformat(),
        },
        "order": order,                 # порядок ключей в исходном глоссарии
        "review_keys": review_keys,      # какие из них проверяются aya (без builtin)
        "result": result,                # текущее состояние терминов (ключ -> term-dict)
        "disagreements": disagreements,  # накопленные расхождения qwen vs aya
    }
    atomic_save(data, CHECKPOINT_FILE)


def load_checkpoint():
    """Возвращает (result, disagreements, order, review_keys, chunks_done, total_chunks)
    или None при невалидном/отсутствующем чекпойнте."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata") or {}
        order = data.get("order") or []
        review_keys = data.get("review_keys") or []
        result = data.get("result") or {}
        disagreements = data.get("disagreements") or []
        chunks_done = int(meta.get("chunks_done", 0))
        total_chunks = int(meta.get("total_chunks", 0))
        if not order or not review_keys or not result:
            return None
        return result, disagreements, order, review_keys, chunks_done, total_chunks
    except Exception as e:
        print(f"  ! Не удалось загрузить чекпойнт ({e}) — старт с нуля")
        return None


def write_disagreements(disagreements, originals, model):
    """Пишет .txt и .json с расхождениями qwen vs aya."""
    with open(DISAGREE_TXT, "w", encoding="utf-8") as f:
        f.write(f"Разногласий: {len(disagreements)}  (qwen → {model})\n")
        f.write("=" * 60 + "\n")
        for k, q, a in disagreements:
            hj = originals.get(k, {}).get("hanja", "")
            f.write(f"{k}{(' ['+hj+']') if hj else ''}\n  qwen: {q}\n  aya : {a}\n\n")

    dis_json = {
        "metadata": {
            "check_model": model,
            "disagreements": len(disagreements),
            "generated_at": datetime.now().isoformat(),
        },
        "disagreements": [
            {
                "korean": k,
                "hanja": originals.get(k, {}).get("hanja", ""),
                "category": originals.get(k, {}).get("category", ""),
                "note": (originals.get(k, {}).get("note", "") or ""),
                "qwen": q,
                "aya": a,
            }
            for k, q, a in disagreements
        ],
    }
    atomic_save(dis_json, DISAGREE_JSON)


def apply_overrides(fixed):
    """Применяет ручные оверрайды из 2_glossary_overrides.json (если есть)."""
    overrides_path = GLOSSARY_OVERRIDES_FILE
    if not os.path.exists(overrides_path):
        return 0
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
        n = 0
        for t in fixed:
            o = ov.get(t.get("korean"))
            if not o:
                continue
            if o.get("russian"):
                t["russian"] = o["russian"]
            if o.get("note"):
                t["note"] = o["note"]
            n += 1
        if n:
            print(f"Применено ручных оверрайдов: {n} (из {overrides_path})")
        return n
    except Exception as e:
        print(f"  ! Оверрайды не применены: {e}")
        return 0


def main():
    p = argparse.ArgumentParser(description="Этап 3: кросс-проверка глоссария")
    p.add_argument("--input", default=GLOSSARY_FILE)
    p.add_argument("--model", default=MODEL_CHECK)
    p.add_argument("--chunk", type=int, default=GLOSSARY_NORMALIZE_CHUNK)
    p.add_argument("--resume", action="store_true",
                   help="Продолжить с чекпойнта (или начать с нуля, если его нет). "
                        "Если стадия уже завершена (есть 2_glossary.checked.json) — выйти без работы.")
    p.add_argument("--force-fresh", action="store_true",
                   help="Игнорировать чекпойнт и done-маркер, начать с нуля от qwen")
    args = p.parse_args()

    # --- 0. Проверка done-маркера ---
    if args.resume and os.path.exists(CHECKED_MARKER) and not args.force_fresh:
        print(f"Стадия уже завершена ранее (есть {CHECKED_MARKER}). Пропускаю.")
        print("  Чтобы перепроверить заново: удали этот файл (и чекпойнт, если есть) "
              "или запусти с --force-fresh.")
        return

    # --- 1. Подготовка qwen-бэкапа (всегда от него стартуем) ---
    if not os.path.exists(QWEN_BACKUP):
        if not os.path.exists(args.input):
            print(f"ОШИБКА: нет ни {QWEN_BACKUP}, ни {args.input}")
            sys.exit(1)
        shutil.copy2(args.input, QWEN_BACKUP)
        print(f"Бэкап qwen-глоссария: {QWEN_BACKUP}")

    with open(QWEN_BACKUP, "r", encoding="utf-8") as f:
        base = json.load(f)
    terms = base.get("terms", [])
    if not terms:
        print("Глоссарий пуст.")
        return

    # --- 2. Оригиналы (источник истины для korean/romanization/hanja) ---
    originals = {}
    order = []
    for t in terms:
        k = (t.get("korean") or "").strip()
        if not k or k in originals:
            continue
        originals[k] = t
        order.append(k)

    review_keys = [k for k in order if not originals[k].get("builtin")]
    builtin_keys = [k for k in order if originals[k].get("builtin")]
    total_chunks = (len(review_keys) + args.chunk - 1) // args.chunk if review_keys else 0

    # --- 3. Resume или старт с нуля ---
    result = {k: dict(originals[k]) for k in order}
    disagreements = []
    chunks_done = 0

    if args.force_fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"--force-fresh: чекпойнт удалён, стартую с нуля")

    if args.resume and not args.force_fresh:
        cp = load_checkpoint()
        if cp:
            cp_result, cp_dis, cp_order, cp_review, cp_done, cp_total = cp
            # Защита: чекпойнт должен соответствовать текущему глоссарию (тот же order)
            if cp_order == order and cp_review == review_keys and cp_total == total_chunks:
                result = {k: dict(originals[k]) for k in order}
                # накладываем уже сделанные правки aya
                for k, v in cp_result.items():
                    if k in result:
                        # ключ/hanja/romanization из originals не трогаем
                        for fld in ("russian", "category", "note"):
                            if fld in v and v.get(fld):
                                result[k][fld] = v[fld]
                disagreements = [tuple(d) if isinstance(d, list) else d for d in cp_dis]
                chunks_done = cp_done
                print(f"Resume: чекпойнт найден, продолжаю с чанка {chunks_done+1}/{total_chunks}")
                print(f"  Уже накоплено разногласий: {len(disagreements)}")
            else:
                print("Resume: чекпойнт несовместим с текущим глоссарием — игнорирую и стартую с нуля")
                chunks_done = 0
                disagreements = []
        else:
            print("Resume: чекпойнта нет, стартую с нуля")

    if builtin_keys:
        print(f"Builtin-гоноративы (без проверки): {len(builtin_keys)}")
    print(f"Проверка глоссария: {len(review_keys)} терминов, модель {args.model}")
    print(f"База (чистый qwen): {QWEN_BACKUP}")
    print(f"Чанки: {chunks_done}/{total_chunks} (chunk size = {args.chunk})")
    print("-" * 60)

    # --- 4. Прогон чанков ---
    for chunk_idx in range(chunks_done, total_chunks):
        ci = chunk_idx * args.chunk
        keys = review_keys[ci:ci + args.chunk]
        compact = [{"korean": k,
                    "hanja": originals[k].get("hanja", ""),
                    "russian": result[k].get("russian"),
                    "category": result[k].get("category"),
                    "note": (result[k].get("note", "") or "")[:80]} for k in keys]
        n = chunk_idx + 1
        print(f"  Чанк {n}/{total_chunks} ({len(keys)})...", end=" ", flush=True)
        resp = call_ollama(CHECK_SYSTEM, json.dumps(compact, ensure_ascii=False, indent=2), args.model)
        got = parse_json(resp).get("terms", []) if resp else []
        applied = 0
        for t in got:
            if not isinstance(t, dict):
                continue
            k = (t.get("korean") or "").strip()
            if k not in originals:
                continue
            if originals[k].get("builtin"):
                continue
            new_ru = (t.get("russian") or "").strip()
            if new_ru and new_ru != (result[k].get("russian") or "").strip():
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

        # СОХРАНЯЕМ ЧЕКПОЙНТ после каждого чанка
        save_checkpoint(result, disagreements, order, review_keys,
                        chunks_done=chunk_idx + 1, total_chunks=total_chunks,
                        model=args.model)
        time.sleep(0.5)

    # --- 5. Финал: оверрайды, запись 2_glossary.json, done-маркер, disagreements ---
    fixed = [result[k] for k in order]
    apply_overrides(fixed)
    fixed.sort(key=lambda x: (x.get("category", ""), x.get("korean", "")))

    out = {"metadata": {"check_model": args.model, "total_terms": len(fixed),
                        "disagreements": len(disagreements), "source": "cross-check",
                        "completed_at": datetime.now().isoformat()},
           "terms": fixed}
    atomic_save(out, args.input)

    write_disagreements(disagreements, originals, args.model)

    # done-маркер: копия финального глоссария (используется bat для skip-чека)
    shutil.copy2(args.input, CHECKED_MARKER)

    # чекпойнт больше не нужен
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\nГотово. {args.input} обновлён (korean/hanja-ключи сохранены от qwen).")
    print(f"Done-маркер:  {CHECKED_MARKER}")
    print(f"Разногласий:  {len(disagreements)} → {DISAGREE_TXT}")
    print(f"                                       {DISAGREE_JSON}")
    print("Дальше: python gemini_fallback.py --mode glossary  (арбитр разногласий)")
    print("Или сразу: python stage4_translate.py")


if __name__ == "__main__":
    main()
