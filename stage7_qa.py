"""
Этап 7: Контроль качества (QA).

Чисто оффлайн-проверка БЕЗ обращения к Ollama. Читает результат вычитки
(4_final.json) или, если его нет, черновой перевод (3_translated.json),
сверяется с глоссарием и помечает сегменты, которые стоит проверить руками.

Что проверяется:
  1. Соблюдение глоссария — если корейский термин есть в оригинале сегмента,
     должен ли его русский эквивалент (с учётом склонения) встретиться в переводе.
  2. Остатки корейского — доля корейских символов в русском тексте.
  3. Аномалии длины — перевод подозрительно короче/длиннее оригинала.
  4. Проблемы из прошлых стадий — ошибки перевода, fallback вычитки, пропуски.
  5. Грубые повторы — один и тот же абзац продублирован внутри сегмента.

Результат: JSON-отчёт (5_qa_report.json) + человекочитаемый .txt рядом с ним.

Запуск:
  python stage7_qa.py
  python stage7_qa.py --source translated     # проверять черновой перевод
  python stage7_qa.py --only-flagged          # в txt только проблемные сегменты
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

from config import *


# ============================================================
# ХЕЛПЕРЫ
# ============================================================

def korean_char_count(text: str) -> int:
    return sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)


def cyrillic_char_count(text: str) -> int:
    return sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)


# Слова, которые сами по себе ничего не значат как «индикатор имени»
_GENERIC_RU = {
    "королевство", "империя", "город", "деревня", "гора", "горы", "река",
    "море", "лес", "замок", "храм", "орден", "клан", "семья", "род", "дом",
    "граф", "графиня", "барон", "герцог", "король", "королева", "принц",
    "принцесса", "святой", "магия", "техника", "навык", "заклинание",
    "меч", "артефакт", "предмет", "зверь", "демон", "монстр", "бог", "богиня",
}


def russian_indicators(russian: str) -> list:
    """Из русского перевода термина достаём «опорные» основы слов, по которым
    можно понять, что термин реально использован в переводе (с учётом склонения).

    Приоритет — слова с заглавной буквы (имена собственные). Если таких нет —
    берём самое длинное значимое слово. Возвращаем основы (усечённые), чтобы
    ловить склонённые формы: «Юдер» → «юдер» найдётся в «Юдера», «Юдеру».
    """
    if not russian:
        return []
    words = re.findall(r"[А-Яа-яЁё]+", russian)
    words = [w for w in words if len(w) >= 4]
    if not words:
        return []

    proper = [w for w in words if w[0].isupper() and w.lower() not in _GENERIC_RU]
    candidates = proper if proper else [max(words, key=len)]

    stems = []
    for w in candidates:
        wl = w.lower()
        # усекаем окончание сильнее, чтобы ловить склонения коротких имён
        # («Майя»→«май» найдётся в «Майе/Майю», «Плеяды»→«пле» в «Плеядах»); минимум 3
        stem = wl[:max(3, len(wl) - 3)]
        stems.append(stem)
    return stems


def find_repeated_block(text: str) -> str:
    """Грубая эвристика: ищем длинный (>=60 симв) абзац, который встречается
    в тексте больше одного раза — частый признак залипания модели."""
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) >= 60]
    seen = set()
    for p in paras:
        if p in seen:
            return p[:80]
        seen.add(p)
    return ""


# ============================================================
# ПРОВЕРКА ОДНОГО СЕГМЕНТА
# ============================================================

def check_segment(item: dict, text_key: str, glossary_terms: list) -> dict:
    seg_id = item.get("id")
    source = item.get("source", "") or ""
    translation = item.get(text_key) or item.get("draft_translation") \
        or item.get("translation") or ""

    flags = []

    # --- проблемы из прошлых стадий ---
    if item.get("error"):
        flags.append({"type": "translation_error",
                      "detail": item.get("error_reason", "")})
    if item.get("skipped"):
        flags.append({"type": "skipped",
                      "detail": item.get("skip_reason", "")})
    if item.get("fallback_reason"):
        flags.append({"type": "proofread_fallback",
                      "detail": item.get("fallback_reason", "")})

    tlen = len(translation)
    slen = len(source)

    # --- пустой/слишком короткий ---
    if tlen < 10:
        flags.append({"type": "empty_or_tiny", "detail": f"{tlen} симв"})
        return _result(seg_id, slen, tlen, flags, translation)

    # --- остатки корейского ---
    kc = korean_char_count(translation)
    if tlen > 0 and kc / tlen > QA_KOREAN_RATIO_WARN:
        flags.append({"type": "korean_leftover",
                      "detail": f"{kc} корейских симв ({kc/tlen:.1%})"})

    # --- мало кириллицы ---
    cc = cyrillic_char_count(translation)
    if tlen > 0 and cc / tlen < 0.30:
        flags.append({"type": "low_cyrillic",
                      "detail": f"{cc}/{tlen} ({cc/tlen:.1%})"})

    # --- аномалии длины ---
    # Для очень коротких сегментов соотношение длин шумное (корейский в символах
    # компактнее русского), поэтому проверяем только содержательные сегменты.
    if slen >= 40:
        ratio = tlen / slen
        if ratio < QA_LEN_RATIO_LOW:
            flags.append({"type": "too_short",
                          "detail": f"перевод/оригинал = {ratio:.2f}"})
        elif ratio > QA_LEN_RATIO_HIGH:
            flags.append({"type": "too_long",
                          "detail": f"перевод/оригинал = {ratio:.2f}"})

    # --- грубый повтор ---
    rep = find_repeated_block(translation)
    if rep:
        flags.append({"type": "repeated_block", "detail": rep})

    # --- соблюдение глоссария ---
    missing_terms = []
    translation_lower = translation.lower()
    for t in glossary_terms:
        kor = (t.get("korean") or "").strip()
        rus = (t.get("russian") or "").strip()
        if not kor or not rus:
            continue
        if kor not in source:
            continue
        stems = russian_indicators(rus)
        if not stems:
            continue
        if not any(stem in translation_lower for stem in stems):
            missing_terms.append({"korean": kor, "russian": rus,
                                  "category": t.get("category", "")})
    if missing_terms:
        flags.append({"type": "glossary_miss",
                      "detail": f"{len(missing_terms)} терм.",
                      "terms": missing_terms})

    return _result(seg_id, slen, tlen, flags, translation)


def _result(seg_id, slen, tlen, flags, translation):
    return {
        "id": seg_id,
        "source_len": slen,
        "translation_len": tlen,
        "flags": flags,
        "preview": translation[:120].replace("\n", " "),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Этап 7: QA-проверка перевода")
    parser.add_argument("--source", choices=["final", "translated"], default="final")
    parser.add_argument("--final", type=str, default=FINAL_FILE)
    parser.add_argument("--translated", type=str, default=TRANSLATED_FILE)
    parser.add_argument("--glossary", type=str, default=GLOSSARY_FILE)
    parser.add_argument("--output", type=str, default=QA_REPORT_FILE)
    parser.add_argument("--only-flagged", action="store_true",
                        help="В txt-отчёт писать только проблемные сегменты")
    args = parser.parse_args()

    # --- выбираем источник ---
    if args.source == "final" and os.path.exists(args.final):
        path = args.final
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("results", [])
        text_key = "final_translation"
    elif os.path.exists(args.translated):
        path = args.translated
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("translations", [])
        text_key = "translation"
    else:
        print("ОШИБКА: Не найден ни 4_final.json, ни 3_translated.json")
        sys.exit(1)

    if not items:
        print(f"ОШИБКА: В {path} нет сегментов")
        sys.exit(1)

    glossary_terms = []
    if os.path.exists(args.glossary):
        with open(args.glossary, "r", encoding="utf-8") as f:
            glossary_terms = json.load(f).get("terms", [])

    print(f"Источник: {path}")
    print(f"Сегментов: {len(items)} | Глоссарий: {len(glossary_terms)} терминов")
    print("-" * 60)

    seg_reports = []
    for item in sorted(items, key=lambda x: x.get("id", 0)):
        seg_reports.append(check_segment(item, text_key, glossary_terms))

    flagged = [r for r in seg_reports if r["flags"]]

    # --- агрегированная статистика ---
    flag_counts = {}
    missing_term_counts = {}
    for r in flagged:
        for fl in r["flags"]:
            flag_counts[fl["type"]] = flag_counts.get(fl["type"], 0) + 1
            if fl["type"] == "glossary_miss":
                for mt in fl.get("terms", []):
                    key = f"{mt['korean']} → {mt['russian']}"
                    missing_term_counts[key] = missing_term_counts.get(key, 0) + 1

    top_missing = sorted(missing_term_counts.items(), key=lambda x: -x[1])[:25]

    report = {
        "metadata": {
            "source_file": path,
            "segments_total": len(seg_reports),
            "segments_flagged": len(flagged),
            "flag_counts": flag_counts,
            "generated_at": datetime.now().isoformat(),
        },
        "top_missing_glossary_terms": [
            {"term": k, "missed_in_segments": v} for k, v in top_missing
        ],
        "segments": seg_reports,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- человекочитаемый txt ---
    txt_path = args.output.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ОТЧЁТ QA\n")
        f.write("=" * 60 + "\n")
        f.write(f"Источник: {path}\n")
        f.write(f"Всего сегментов: {len(seg_reports)}\n")
        f.write(f"С замечаниями: {len(flagged)} "
                f"({len(flagged)/max(len(seg_reports),1):.1%})\n\n")

        f.write("Сводка по типам замечаний:\n")
        if flag_counts:
            for k, v in sorted(flag_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {k}: {v}\n")
        else:
            f.write("  (замечаний нет)\n")
        f.write("\n")

        if top_missing:
            f.write("Чаще всего «теряются» термины глоссария:\n")
            for k, v in top_missing:
                f.write(f"  [{v}x] {k}\n")
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("СЕГМЕНТЫ\n")
        f.write("-" * 60 + "\n")
        to_show = flagged if args.only_flagged else seg_reports
        for r in to_show:
            if not r["flags"]:
                continue
            f.write(f"\n#{r['id']}  (ориг {r['source_len']} → пер {r['translation_len']} симв)\n")
            for fl in r["flags"]:
                line = f"  ⚠ {fl['type']}: {fl.get('detail', '')}"
                f.write(line + "\n")
                if fl["type"] == "glossary_miss":
                    for mt in fl.get("terms", [])[:12]:
                        f.write(f"      · {mt['korean']} → {mt['russian']}\n")
            f.write(f"  «{r['preview']}»\n")

    # --- консоль ---
    print(f"С замечаниями: {len(flagged)}/{len(seg_reports)} сегментов")
    if flag_counts:
        for k, v in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    if top_missing:
        print("\nЧаще всего теряются термины:")
        for k, v in top_missing[:10]:
            print(f"  [{v}x] {k}")
    print(f"\nОтчёт JSON: {args.output}")
    print(f"Отчёт TXT:  {txt_path}")


if __name__ == "__main__":
    main()
