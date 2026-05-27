"""
Облачный фолбэк через Gemini — РУЧНОЙ добор того, что локальные модели не осилили.

Запускается отдельно, когда qwen/aya/gemma на части сегментов выдали мусор
(остатки корейского, мало кириллицы, петли повторов, [ОШИБКА ПЕРЕВОДА]) или
когда ты сам знаешь номер проблемного сегмента.

Что делает:
  1. Проходит по выводу пайплайна и ищет проблемные сегменты (или берёт --ids).
  2. Шлёт проблемное место в облачный Gemini (перевод или полировка).
  3. Подставляет ответ на место сегмента.
  4. Помечает, что перевод выполнил Gemini (model + translated_by="gemini" +
     cloud_fallback=true), прежнюю битую попытку сохраняет в failed_translation.

Ключ: apikey.txt рядом со скриптом, либо env GEMINI_API_KEY, либо --api-key.
Пакет: pip install google-genai

Примеры:
  python gemini_fallback.py --dry-run                 # показать кандидатов (без вызовов)
  python gemini_fallback.py                           # перевести все автонайденные проблемные
  python gemini_fallback.py --ids 6,25                # добить конкретные сегменты
  python gemini_fallback.py --ids 6 --also-final      # ещё и обновить 4_final.json
  python gemini_fallback.py --mode polish --ids 14    # перешлифовать сегмент в 4_final.json
  python gemini_fallback.py --mode glossary           # арбитраж разногласий qwen/aya в глоссарии
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime

from config import *

# Переиспользуем подбор релевантных терминов глоссария из переводчика
try:
    from stage4_translate import find_relevant_terms, format_glossary_for_prompt
except Exception:  # на случай переименований — мягкая деградация без глоссария
    def find_relevant_terms(terms, text, max_terms=60):
        return [t for t in terms if (t.get("korean") or "") in text][:max_terms]

    def format_glossary_for_prompt(terms):
        if not terms:
            return ""
        return "\n".join(
            f"  {t.get('korean')}"
            + (f" ({t['hanja']})" if t.get("hanja") else "")
            + f" -> {t.get('russian')}"
            for t in terms
        )


# ============================================================
# ПРОМПТЫ
# ============================================================

TRANSLATE_SYSTEM = (
    "Ты - профессиональный литературный переводчик с корейского на русский "
    "(ранобэ/веб-новеллы). Этот фрагмент локальная модель не смогла перевести "
    "корректно - часто это игровой чат с мусором, повторами и цензурой. Переведи "
    "его ПОЛНОСТЬЮ и аккуратно.\n\n"
    "ПРАВИЛА:\n"
    "- Переводи смысл на живой русский; корейский игровой/сетевой сленг - русским "
    "сленгом (ㅋㅋㅋ -> «ахах», 멘붕 -> «поплыл/подгорело», 고인물/썩은물 -> «олд/задрот»).\n"
    "- Формат «Ник : реплика»: НИК оставляй как есть (латиница/цифры; если он есть в "
    "глоссарии - по глоссарию), переводи только реплику. Двоеточие сохраняй.\n"
    "- Кейсмэш и случайные буквы (напр. sdlkghiosd...) оставляй БЕЗ изменений.\n"
    "- ПОВТОРЯЮЩИЕСЯ строки сохраняй ровно столько же раз, сколько в оригинале.\n"
    "- Цензуру символом | сохраняй (ㅆ|발 -> «бл|ть»).\n"
    "- Числа переноси точно. Имена/термины - строго по глоссарию (он в сообщении; "
    "если указана ханча - переводи по смыслу иероглифов, не фонетикой).\n"
    "- Сохраняй разбивку по строкам и абзацам.\n"
    "- Не добавляй ничего, чего нет в оригинале.\n"
    "- Выводи ТОЛЬКО русский перевод, без пояснений и префиксов."
)

POLISH_SYSTEM = (
    "Ты - литературный редактор русского текста (перевод корейского ранобэ).\n"
    "Улучшай ТОЛЬКО форму: живой естественный русский, грамматика, пунктуация, ритм.\n"
    "НЕ меняй смысл, не добавляй и не убирай факты, события, реплики, имена, числа.\n"
    "Имена и термины - строго по глоссарию (он в сообщении). Сохраняй разбивку на "
    "абзацы и реплики.\n"
    "Выводи ТОЛЬКО отредактированный русский текст, без пояснений и префиксов."
)

GLOSSARY_ARBITER_SYSTEM = (
    "Ты - строгий билингвальный судья перевода КОРЕЙСКОЙ ЛЕКСИКИ на русский для "
    "ранобэ/веб-новелл (часто уся-фэнтези). Тебе дают спорный термин и ДВА варианта "
    "русского перевода - от модели A (qwen) и модели B (aya). Твоя задача - выбрать "
    "правильный или, если оба плохи, предложить третий.\n\n"
    "ПРАВИЛА (применяй СТРОГО):\n"
    "1. КОНЦЕПТ-ТЕРМИНЫ С ХАНЧА (поле hanja заполнено): перевод СТРОИТСЯ ПО СМЫСЛУ "
    "иероглифов, НЕ фонетикой корейского чтения. Голая фонетическая транслитерация "
    "такого термина - ОШИБКА.\n"
    "   九陰絶脈 → «меридианы Девяти Инь»   (НЕ «Гуин Цзюймэй»)\n"
    "   天武之體 → «тело Небесного Воина»   (НЕ «Чхонму Чхи»)\n"
    "   九劍 → «Девять Мечей»              (НЕ «Гу Гом»)\n"
    "2. ЗАИМСТВОВАНИЯ из английского/иностранных языков (часто это видно по корейской "
    "транслитерации - ㅍ=F, ㄹ=L): восстанавливай оригинал, а не транслитерируй вслепую.\n"
    "   플라이 → Fly (полёт),  아웃복서 → Outboxer / Аутбоксер,\n"
    "   프로스트 앤빌 → Frost Anvil,  카플란 → Kaplan.\n"
    "3. ИМЕНА И МЕСТА: транслитерация по системе Концевича, не китайский пиньинь.\n"
    "   강진호 - «Кан Джинхо» (НЕ «Ган Жэньхао»). Без дефисов в именах.\n"
    "4. ТИТУЛЫ (фиксированный маппинг): 공작=герцог, 대공=великий герцог, 후작=маркиз,\n"
    "   변경백=маркграф, 백작=граф, 자작=виконт, 남작=барон. 백작가 = «графский род» "
    "(НЕ барон).\n"
    "5. ОМОНИМЫ ХАНЧА: выбирай значение по смыслу контекста (변경 邊境 граница vs 變更 "
    "перемена; 성 聖 святой vs 城 крепость).\n"
    "6. ЧИСЛА В ХАНЧА: 一=1, 二=2, 三=3, 九=9, 十=10 - никогда не путай.\n"
    "7. Не выдумывай ханча, не меняй корейский ключ. Только русская сторона.\n\n"
    "ФОРМАТ ОТВЕТА: СТРОГО один JSON-объект, без пояснений вне JSON, без markdown:\n"
    "{\"choice\": \"qwen\" | \"aya\" | \"other\", "
    "\"russian\": \"итоговый русский перевод\", "
    "\"reason\": \"кратко почему (1 фраза)\"}\n\n"
    "- choice=\"qwen\" - вариант A правильный.\n"
    "- choice=\"aya\"  - вариант B правильный.\n"
    "- choice=\"other\" - оба плохи, в поле russian свой вариант.\n"
    "В поле russian ВСЕГДА пиши финальный русский перевод (даже если choice=qwen/aya - "
    "продублируй выбранный текст)."
)


# ============================================================
# GEMINI
# ============================================================

def read_api_key(args) -> str:
    if args.api_key:
        return args.api_key.strip()
    env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env:
        return env.strip()
    path = args.api_key_file or APIKEY_FILE
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip().strip('"').strip("'").strip()
        if key:
            return key
    print(f"ОШИБКА: не найден API-ключ. Положи его в {APIKEY_FILE}, "
          f"или задай env GEMINI_API_KEY, или передай --api-key.")
    sys.exit(1)


def make_client(api_key):
    try:
        from google import genai
    except ImportError:
        print("ОШИБКА: пакет не установлен. Установи: pip install google-genai")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def call_gemini(client, model, prompt, temperature=0.3, max_retries=3):
    last = ""
    for attempt in range(1, max_retries + 1):
        try:
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config={"temperature": temperature},
                )
            except TypeError:
                # старые версии SDK без config-параметра
                resp = client.models.generate_content(model=model, contents=prompt)
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            last = str(e)
            print(f"  ! Gemini ошибка ({attempt}/{max_retries}): {e}")
            time.sleep(5)
    print(f"  ! Не удалось получить ответ Gemini: {last}")
    return ""


def clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[\w]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"^(Перевод|Translation|РУССКИЙ ПЕРЕВОД|Отредактированный текст"
                  r"|Результат|Вот перевод)[:：]?\s*\n?", "", text, flags=re.IGNORECASE)
    return text.strip()


# ============================================================
# ДЕТЕКТОР ПРОБЛЕМ
# ============================================================

def korean_ratio(t):
    if not t:
        return 0.0
    return sum(1 for c in t if 0xAC00 <= ord(c) <= 0xD7AF) / len(t)


def cyrillic_ratio(t):
    if not t:
        return 0.0
    return sum(1 for c in t if 0x0400 <= ord(c) <= 0x04FF) / len(t)


def diagnose_translation(item) -> str:
    """Возвращает причину проблемы (str) или '' если сегмент в порядке."""
    if item.get("error"):
        return "error:" + str(item.get("error_reason", ""))
    tr = (item.get("translation") or "").strip()
    if not tr or "[ОШИБКА ПЕРЕВОДА]" in tr:
        return "placeholder/empty"
    if len(tr) < 10:
        return "too_short(" + str(len(tr)) + ")"
    if korean_ratio(tr) > 0.20:
        return "korean_leftover(" + format(korean_ratio(tr), ".0%") + ")"
    if cyrillic_ratio(tr) < 0.20:
        return "low_cyrillic(" + format(cyrillic_ratio(tr), ".0%") + ")"
    return ""


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def atomic_save(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def backup(path):
    bak = os.path.join(BACKUP_DIR, os.path.basename(path)
                       + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    shutil.copy2(path, bak)
    print("Бэкап: " + bak)


def parse_ids(s):
    if not s:
        return set()
    out = set()
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


# ============================================================
# РЕЖИМ TRANSLATE (3_translated.json)
# ============================================================

def run_translate(args, client, glossary):
    path = args.input or TRANSLATED_FILE
    if not os.path.exists(path):
        print("ОШИБКА: нет файла " + path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["translations"]
    by_id = {t["id"]: t for t in items}

    forced = parse_ids(args.ids)
    if forced:
        targets = [(sid, "forced(--ids)") for sid in sorted(forced) if sid in by_id]
        missing = [sid for sid in forced if sid not in by_id]
        if missing:
            print("  ! нет таких id в файле: " + str(sorted(missing)))
    else:
        targets = []
        for t in sorted(items, key=lambda x: x["id"]):
            reason = diagnose_translation(t)
            if not reason and args.include_check_failed and t.get("check_failed"):
                reason = "check_failed:" + str(t.get("check_fail_reason", ""))
            if reason:
                targets.append((t["id"], reason))

    if args.limit > 0:
        targets = targets[:args.limit]

    if not targets:
        print("Проблемных сегментов не найдено (или все отфильтрованы). Нечего делать.")
        return

    print("Режим TRANSLATE | файл: " + path)
    print("К отправке в Gemini (" + args.model + "): " + str(len(targets)) + " сегментов")
    for sid, why in targets:
        src_preview = (by_id[sid].get("source", "") or "")[:60].replace("\n", " ")
        print("  #" + str(sid) + "  [" + why + "]  <" + src_preview + ">")

    if args.dry_run:
        print("\n--dry-run: вызовов Gemini не было, файл не изменён.")
        return

    if not args.no_backup:
        backup(path)

    fixed = 0
    for i, (sid, why) in enumerate(targets):
        it = by_id[sid]
        src = it.get("source", "")
        rel = find_relevant_terms(glossary, src) if glossary else []
        gloss = format_glossary_for_prompt(rel) if rel else ""
        gloss_block = ("ГЛОССАРИЙ (строго):\n" + gloss + "\n\n") if gloss else ""
        prompt = (TRANSLATE_SYSTEM + "\n\n" + gloss_block
                  + "КОРЕЙСКИЙ ТЕКСТ (сегмент #" + str(sid) + "):\n---\n" + src
                  + "\n---\n\nПЕРЕВОД (только русский текст):")
        print("[" + str(i + 1) + "/" + str(len(targets)) + "] #" + str(sid)
              + " (" + str(len(src)) + " симв) -> Gemini...", flush=True)
        out = clean(call_gemini(client, args.model, prompt, args.temperature))
        if not out or len(out) < 5:
            print("  ! пустой ответ Gemini - сегмент не тронут")
            continue
        if korean_ratio(out) > 0.30:
            print("  ! предупреждение: в ответе много корейского ("
                  + format(korean_ratio(out), ".0%") + ") - подставляю, но проверь руками")

        it["failed_translation"] = it.get("translation", "")
        it["translation"] = out
        it["model"] = args.model
        it["translated_by"] = "gemini"
        it["cloud_fallback"] = True
        it["checked"] = True
        it["check_model"] = "gemini-fallback"
        for k in ("error", "error_reason", "check_failed", "check_fail_reason"):
            it.pop(k, None)
        fixed += 1
        preview = out[:90].replace("\n", " ")
        print("  OK (" + str(len(out)) + " симв) -> " + preview + ("..." if len(out) > 90 else ""))

        data.setdefault("metadata", {})["cloud_fallback_last"] = datetime.now().isoformat()
        atomic_save(data, path)
        if i < len(targets) - 1:
            time.sleep(args.delay)

    print("\nГотово. Заменено Gemini: " + str(fixed) + "/" + str(len(targets)) + ". Файл: " + path)

    if args.also_final and fixed:
        sync_to_final([sid for sid, _ in targets], by_id, args)


def sync_to_final(ids, by_id, args):
    """Перенести свежие переводы Gemini в 4_final.json (чтобы не гонять stage6
    ради пары сегментов). Обновляет final_translation и draft_translation."""
    fpath = FINAL_FILE
    if not os.path.exists(fpath):
        print("  --also-final: " + fpath + " не найден, пропуск")
        return
    with open(fpath, "r", encoding="utf-8") as f:
        fdata = json.load(f)
    fres = {r["id"]: r for r in fdata.get("results", [])}
    if not args.no_backup:
        backup(fpath)
    n = 0
    for sid in ids:
        it = by_id.get(sid)
        r = fres.get(sid)
        if not it or not r or not it.get("cloud_fallback"):
            continue
        new = it.get("translation", "")
        r["draft_translation"] = new
        r["final_translation"] = new
        r["cloud_fallback"] = True
        r["polished_by"] = "gemini-fallback (без полировки)"
        r.pop("fallback_reason", None)
        r.pop("skipped", None)
        r.pop("skip_reason", None)
        n += 1
    atomic_save(fdata, fpath)
    print("  --also-final: обновлено сегментов в " + fpath + ": " + str(n))


# ============================================================
# РЕЖИМ POLISH (4_final.json)
# ============================================================

def run_polish(args, client, glossary):
    path = args.input or FINAL_FILE
    if not os.path.exists(path):
        print("ОШИБКА: нет файла " + path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    by_id = {r["id"]: r for r in results}

    forced = parse_ids(args.ids)
    if forced:
        targets = [(sid, "forced(--ids)") for sid in sorted(forced) if sid in by_id]
    else:
        # без --ids в polish берём те, что упали в fallback вычитки
        targets = [(r["id"], "fallback:" + str(r.get("fallback_reason", "")))
                   for r in sorted(results, key=lambda x: x["id"])
                   if r.get("fallback_reason")]

    if args.limit > 0:
        targets = targets[:args.limit]

    if not targets:
        print("Нет кандидатов на полировку (нет --ids и нет fallback'ов).")
        return

    print("Режим POLISH | файл: " + path)
    print("К отправке в Gemini (" + args.model + "): " + str(len(targets)) + " сегментов")
    for sid, why in targets:
        print("  #" + str(sid) + "  [" + why + "]")

    if args.dry_run:
        print("\n--dry-run: вызовов Gemini не было, файл не изменён.")
        return

    if not args.no_backup:
        backup(path)

    fixed = 0
    for i, (sid, why) in enumerate(targets):
        r = by_id[sid]
        src = r.get("source", "")
        cur = r.get("final_translation") or r.get("draft_translation", "")
        rel = find_relevant_terms(glossary, src) if glossary else []
        gloss = format_glossary_for_prompt(rel) if rel else ""
        gloss_block = ("ГЛОССАРИЙ (строго):\n" + gloss + "\n\n") if gloss else ""
        prompt = (POLISH_SYSTEM + "\n\n" + gloss_block
                  + "РУССКИЙ ТЕКСТ:\n---\n" + cur + "\n---\n\nОтредактированный текст:")
        print("[" + str(i + 1) + "/" + str(len(targets)) + "] #" + str(sid)
              + " (" + str(len(cur)) + " симв) -> Gemini...", flush=True)
        out = clean(call_gemini(client, args.model, prompt, args.temperature))
        if not out or len(out) < 5:
            print("  ! пустой ответ Gemini - сегмент не тронут")
            continue
        r["draft_before_cloud"] = cur
        r["final_translation"] = out
        r["cloud_fallback"] = True
        r["polished_by"] = "gemini"
        r.pop("fallback_reason", None)
        fixed += 1
        preview = out[:90].replace("\n", " ")
        print("  OK (" + str(len(out)) + " симв) -> " + preview + ("..." if len(out) > 90 else ""))
        data.setdefault("metadata", {})["cloud_fallback_last"] = datetime.now().isoformat()
        atomic_save(data, path)
        if i < len(targets) - 1:
            time.sleep(args.delay)

    print("\nГотово. Перешлифовано Gemini: " + str(fixed) + "/" + str(len(targets)) + ". Файл: " + path)


# ============================================================
# РЕЖИМ GLOSSARY (арбитраж разногласий qwen/aya)
# ============================================================

def parse_arbiter_json(text: str):
    """Извлекает {choice/russian/reason} из ответа Gemini.
    Возвращает либо валидный dict, либо (None, причина_отказа) для аудита."""
    if not text:
        return None, "пустой ответ"
    t = re.sub(r"```\w*\n?", "", text).replace("```", "")
    # Берём ПОСЛЕДНИЙ JSON-объект — это страховка от пояснений модели до итогового JSON.
    matches = list(re.finditer(r"\{[\s\S]*?\}", t))
    if not matches:
        return None, "нет JSON-объекта"
    obj = None
    for m in reversed(matches):
        try:
            obj = json.loads(m.group())
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(obj, dict):
        return None, "не словарь"
    choice = str(obj.get("choice", "")).strip().lower()
    if choice not in ("qwen", "aya", "other"):
        return None, f"choice='{choice}' не из {{qwen,aya,other}}"
    russian = str(obj.get("russian", "")).strip()
    if not russian:
        return None, "пустой russian"
    # --- САНИТАРНЫЕ ПРОВЕРКИ СОДЕРЖАНИЯ ---
    # 1. Длина: один глоссарный термин - короткая строка. Длиннее 200 симв = мусор.
    if len(russian) > 200:
        return None, f"слишком длинно ({len(russian)} симв)"
    # 2. Запрет переноса строк и явной чуши: если есть \n - модель сорвалась в текст.
    if "\n" in russian:
        return None, "перевод содержит перенос строки (не одиночный термин)"
    # 3. Запрет корейских символов в финальном русском переводе.
    kor = sum(1 for c in russian if 0xAC00 <= ord(c) <= 0xD7AF)
    if kor > 0:
        return None, f"в переводе остались корейские символы ({kor})"
    # 4. Минимум буквенных символов (кириллица+латиница; латиница нужна для
    #    заимствований типа Fly, Frost Anvil). Без букв - явный мусор.
    letters = sum(1 for c in russian if c.isalpha())
    if letters < 2:
        return None, "почти нет букв"
    # 5. Если строка слишком похожа на ханча/китайские иероглифы - тоже отбой.
    han = sum(1 for c in russian if 0x4E00 <= ord(c) <= 0x9FFF)
    if han > 0 and han * 3 >= letters:
        return None, "в переводе много ханча/кит. иероглифов"
    reason = str(obj.get("reason", "")).strip()[:200]
    return {"choice": choice, "russian": russian, "reason": reason}, ""


def run_glossary(args, client):
    """Арбитраж разногласий qwen vs aya из 2_glossary_disagreements.json.
    Применяет выбор Gemini к 2_glossary.json и пишет лог в .txt."""
    dis_path = os.path.join(OUTPUT_DIR, "2_glossary_disagreements.json")
    if not os.path.exists(dis_path):
        print(f"Нет файла {dis_path}. Запусти stage3_glossary_check сперва.")
        return
    with open(dis_path, "r", encoding="utf-8") as f:
        dis_data = json.load(f)
    disagreements = dis_data.get("disagreements", [])
    if not disagreements:
        print("Разногласий нет - ничего не делаю.")
        return

    # Опциональный фильтр по конкретным корейским ключам
    forced_keys = set()
    if args.ids:
        forced_keys = {s.strip() for s in args.ids.replace(";", ",").split(",") if s.strip()}
    if forced_keys:
        disagreements = [d for d in disagreements if d.get("korean") in forced_keys]
        if not disagreements:
            print(f"Указанных корейских ключей нет в файле разногласий.")
            return

    if args.limit > 0:
        disagreements = disagreements[:args.limit]

    glossary_path = args.input or GLOSSARY_FILE
    if not os.path.exists(glossary_path):
        print(f"ОШИБКА: нет файла глоссария {glossary_path}")
        return
    with open(glossary_path, "r", encoding="utf-8") as f:
        gloss_data = json.load(f)
    by_key = {(t.get("korean") or "").strip(): t for t in gloss_data.get("terms", [])}

    # Идемпотентность: термины, по которым Gemini уже выносил вердикт, пропускаем
    # (если только пользователь явно не указал их через --ids).
    if not forced_keys:
        before = len(disagreements)
        disagreements = [d for d in disagreements
                         if (by_key.get(d.get("korean")) or {}).get("arbiter") != "gemini"]
        skipped = before - len(disagreements)
        if skipped:
            print(f"Пропускаю уже размеченных Gemini: {skipped} (повторный запуск)")
        if not disagreements:
            print("Все разногласия уже разобраны Gemini. Используй --ids для пересмотра.")
            return

    print(f"Режим GLOSSARY (арбитр Gemini)  |  модель: {args.model}")
    print(f"Разногласий к разбору: {len(disagreements)}")
    for d in disagreements:
        hj = (" [" + d["hanja"] + "]") if d.get("hanja") else ""
        print(f"  {d['korean']}{hj}  qwen: {d.get('qwen','')!r}  aya: {d.get('aya','')!r}")

    if args.dry_run:
        print("\n--dry-run: вызовов Gemini не было, файл не изменён.")
        return

    if not args.no_backup:
        backup(glossary_path)

    log_lines = [f"Арбитраж Gemini ({args.model}) по разногласиям qwen vs aya",
                 "=" * 60]
    counts = {"qwen": 0, "aya": 0, "other": 0, "parse_error": 0, "missing": 0}

    for i, d in enumerate(disagreements):
        kor = d["korean"]
        hanja = d.get("hanja", "")
        cat = d.get("category", "")
        note = d.get("note", "")
        q = d.get("qwen", "")
        a = d.get("aya", "")

        # Чтобы быть нейтральным, помечаем варианты как A/B, а соответствие моделям знаем у себя.
        # Случайно не меняем порядок (детерминированно — A=qwen, B=aya), но в промпте подписываем.
        user_lines = [
            f"Korean: {kor}",
        ]
        if hanja:
            user_lines.append(f"Hanja: {hanja}")
        if cat:
            user_lines.append(f"Category: {cat}")
        if note:
            user_lines.append(f"Note: {note[:140]}")
        user_lines.append("")
        user_lines.append(f"Variant A (qwen): {q}")
        user_lines.append(f"Variant B (aya):  {a}")
        user_lines.append("")
        user_lines.append("Выбери правильный или предложи третий. Ответ - один JSON.")
        prompt = GLOSSARY_ARBITER_SYSTEM + "\n\n" + "\n".join(user_lines)

        print(f"[{i+1}/{len(disagreements)}] {kor}"
              f"{(' ['+hanja+']') if hanja else ''} -> Gemini...", flush=True)
        raw = call_gemini(client, args.model, prompt, args.temperature)
        verdict, reject_reason = parse_arbiter_json(raw)
        if not verdict:
            counts["parse_error"] += 1
            print(f"  ! отбой ({reject_reason}) - оставляю значение aya: {a!r}")
            log_lines.append(f"\n{kor}{(' ['+hanja+']') if hanja else ''}")
            log_lines.append(f"  qwen: {q}")
            log_lines.append(f"  aya : {a}")
            log_lines.append(f"  GEMINI: REJECT [{reject_reason}] "
                             f"(raw: {(raw or '')[:120]!r})")
            continue

        target = by_key.get(kor)
        if not target:
            counts["missing"] += 1
            print(f"  ! термина {kor!r} нет в {glossary_path} - пропускаю")
            continue

        # Записываем выбор Gemini
        target["qwen_russian"] = q
        target["aya_russian"] = a
        target["arbiter"] = "gemini"
        target["arbiter_choice"] = verdict["choice"]
        target["arbiter_reason"] = verdict["reason"]
        target["russian"] = verdict["russian"]
        counts[verdict["choice"]] += 1

        log_lines.append(f"\n{kor}{(' ['+hanja+']') if hanja else ''}")
        log_lines.append(f"  qwen: {q}")
        log_lines.append(f"  aya : {a}")
        log_lines.append(f"  GEMINI [{verdict['choice']}]: {verdict['russian']}")
        if verdict["reason"]:
            log_lines.append(f"    причина: {verdict['reason']}")

        preview = (verdict["russian"][:60]).replace("\n", " ")
        print(f"  {verdict['choice'].upper()}: {preview}"
              + ("..." if len(verdict['russian']) > 60 else ""))

        # Сохраняем после каждого вердикта (на случай прерывания)
        gloss_data.setdefault("metadata", {})["arbiter"] = "gemini"
        gloss_data["metadata"]["arbiter_last_run"] = datetime.now().isoformat()
        atomic_save(gloss_data, glossary_path)

        if i < len(disagreements) - 1:
            time.sleep(args.delay)

    # Финальный лог
    log_path = os.path.join(OUTPUT_DIR, "2_glossary_gemini_arbiter.txt")
    log_lines.append("\n" + "=" * 60)
    log_lines.append(f"Итого: qwen={counts['qwen']}  aya={counts['aya']}  "
                     f"other={counts['other']}  parse_error={counts['parse_error']}  "
                     f"missing={counts['missing']}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nГотово. Файл глоссария: {glossary_path}")
    print(f"Лог арбитража:           {log_path}")
    print(f"Итого: qwen={counts['qwen']}  aya={counts['aya']}  other={counts['other']}"
          f"  parse_error={counts['parse_error']}  missing={counts['missing']}")


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Ручной облачный фолбэк через Gemini")
    p.add_argument("--mode", choices=["translate", "polish", "glossary"], default="translate",
                   help="translate: чинить 3_translated.json; polish: перешлифовать 4_final.json; "
                        "glossary: арбитр разногласий qwen/aya в 2_glossary.json")
    p.add_argument("--input", default=None, help="Переопределить входной файл")
    p.add_argument("--glossary", default=GLOSSARY_FILE)
    p.add_argument("--ids", default="", help="Конкретные id (translate/polish) ИЛИ "
                                              "корейские ключи через запятую (glossary)")
    p.add_argument("--include-check-failed", action="store_true",
                   help="(translate) добавить сегменты с check_failed")
    p.add_argument("--also-final", action="store_true",
                   help="(translate) сразу обновить 4_final.json для исправленных id")
    p.add_argument("--model", default=MODEL_CLOUD)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=0, help="Максимум сегментов за прогон (0 = все)")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true", help="Только показать кандидатов, без вызовов")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    # Режим glossary работает с файлом глоссария напрямую — отдельный путь
    if args.mode == "glossary":
        client = None
        if not args.dry_run:
            client = make_client(read_api_key(args))
        run_glossary(args, client)
        return

    glossary = []
    try:
        if os.path.exists(args.glossary):
            with open(args.glossary, "r", encoding="utf-8") as f:
                glossary = json.load(f).get("terms", [])
            print("Глоссарий: " + str(len(glossary)) + " терминов")
        else:
            print("Глоссарий не найден (" + str(args.glossary) + ") - работаю без него")
    except Exception as e:
        print("Глоссарий не загружен (" + str(e) + ") - работаю без него")

    client = None
    if not args.dry_run:
        client = make_client(read_api_key(args))

    if args.mode == "translate":
        run_translate(args, client, glossary)
    else:
        run_polish(args, client, glossary)


if __name__ == "__main__":
    main()
