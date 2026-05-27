"""
Этап 9: Финальная починка по вердикту судьи через Gemini.

Читает 6_judge_report.json (от stage8) и для каждого ПРОБЛЕМНОГО сегмента
посылает в Gemini корейский оригинал + текущий русский перевод + КОНКРЕТНЫЙ
список замечаний судьи. Gemini обязан внести исправления и вернуть финальный
русский текст. Обновляет 4_final.json. НЕ меняет 3_translated.json.

Кого чиним (по умолчанию):
  - verdict == "bad"  (систематические проблемы)
  - verdict == "review" с любой issue severity == "high"
Флаги: --only-bad, --all-review, --ids X,Y,Z

COST GATE:
  - Считаем кандидатов ДО любых вызовов.
  - Если их меньше --max-auto (default 100) — отправляем без вопросов.
  - Если больше:
      * в интерактивном терминале спрашиваем подтверждение (y/N);
      * с перенаправленным выводом (как в run_all.bat) — выходим с ошибкой,
        чтобы юзер запустил вручную с --yes / --limit / --max-auto.

RESUME:
  - Успешные правки помечаются в 4_final.json: judge_fix_applied=true.
  - Сегменты, по которым Gemini прислал мусор, отбиваются валидатором и
    помечаются judge_fix_skipped=<reason> (не повторяются автоматически).
  - При quota/network ошибках НИЧЕГО не помечаем — на следующий запуск
    эти сегменты автоматически попадут в выборку снова.
  - 3 подряд пустых ответа от Gemini → автостоп прогона (вероятно исчерпан
    лимит API). На следующий день --resume продолжит с этого же сегмента.

Запуск:
  python stage9_judge_fix.py                    # с глобальным cost-gate
  python stage9_judge_fix.py --resume           # пропустить уже починенные
  python stage9_judge_fix.py --yes              # послать все без подтверждения
  python stage9_judge_fix.py --limit 50         # не больше 50 за прогон
  python stage9_judge_fix.py --only-bad         # только verdict=bad
  python stage9_judge_fix.py --ids 47,113       # ручной список (перебивает выборку)
  python stage9_judge_fix.py --dry-run          # показать кандидатов без вызовов
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from config import *

# Переиспользуем хелперы Gemini-клиента из gemini_fallback
try:
    from gemini_fallback import (
        read_api_key, make_client, call_gemini, atomic_save, backup,
    )
    from stage4_translate import find_relevant_terms, format_glossary_for_prompt
except Exception as e:
    print(f"ОШИБКА импорта: {e}")
    sys.exit(1)


# ============================================================
# КОНСТАНТЫ
# ============================================================

SEV_RANK = {"high": 3, "medium": 2, "low": 1}
DEFAULT_MAX_AUTO = 100
ABORT_AFTER_CONSECUTIVE_EMPTY = 3
MAX_GLOSSARY_TERMS = 60


# ============================================================
# ПРОМПТ
# ============================================================

JUDGE_FIX_SYSTEM = (
    "Ты - финальный редактор русского литературного перевода с корейского "
    "(ранобэ/веб-новеллы). Тебе дают:\n"
    "  (1) корейский оригинал сегмента,\n"
    "  (2) текущий русский перевод (с проблемами),\n"
    "  (3) КОНКРЕТНЫЙ список замечаний независимого судьи к этому переводу.\n\n"
    "ТВОЯ ЗАДАЧА: внести ВСЕ исправления по списку замечаний и вернуть финальный "
    "русский текст этого сегмента.\n\n"
    "ПРАВИЛА:\n"
    "- Меняй ТОЛЬКО то, на что указал судья. Удачные места не трогай.\n"
    "- Если замечание про лишнее/добавленное/галлюцинацию/дописку — УДАЛИ это "
    "  полностью (а не «смягчи»).\n"
    "- Если замечание про пропуск — добавь по корейскому оригиналу.\n"
    "- Если замечание про неверный смысл слова/термина/числа/имени — исправь "
    "  по оригиналу и глоссарию (если у термина есть ханча — перевод по СМЫСЛУ "
    "  иероглифов, не фонетикой).\n"
    "- Если замечание про корявость/тон — переформулируй сохранив смысл.\n"
    "- НЕ добавляй ничего, чего нет в корейском оригинале (помимо естественных "
    "  русских связок).\n"
    "- Сохраняй разбивку на абзацы и реплики прямой речи; число реплик в твоём "
    "  ответе должно совпадать с числом реплик в корейском оригинале.\n"
    "- Имена и термины — строго по глоссарию.\n"
    "- Выводи ТОЛЬКО исправленный русский текст. Без пояснений, заголовков, "
    "  префиксов и markdown."
)


# ============================================================
# ВЫБОРКА
# ============================================================

def should_fix(judgment, mode):
    """Возвращает True, если сегмент попадает в выборку по режиму."""
    if not judgment:
        return False  # parse_error / пустой — не трогаем
    verdict = (judgment.get("verdict") or "ok").lower()
    issues = judgment.get("issues") or []
    if mode == "only-bad":
        return verdict == "bad"
    if mode == "all-review":
        return verdict in ("bad", "review")
    # default: bad ИЛИ review с high severity
    if verdict == "bad":
        return True
    if verdict == "review" and any((i or {}).get("severity") == "high" for i in issues):
        return True
    return False


def parse_ids(s):
    out = set()
    if not s:
        return out
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


# ============================================================
# ПРОМПТ-СБОРКА
# ============================================================

def build_user_prompt(src, current, issues, glossary_block):
    lines = []
    if glossary_block:
        lines.append(glossary_block)
    lines.append("КОРЕЙСКИЙ ОРИГИНАЛ:")
    lines.append("---")
    lines.append(src)
    lines.append("---")
    lines.append("")
    lines.append("ТЕКУЩИЙ РУССКИЙ ПЕРЕВОД (нужно починить):")
    lines.append("---")
    lines.append(current)
    lines.append("---")
    lines.append("")
    lines.append("ЗАМЕЧАНИЯ СУДЬИ К ЭТОМУ ПЕРЕВОДУ (по серьёзности сверху вниз):")
    sorted_issues = sorted(issues, key=lambda x: -SEV_RANK.get((x or {}).get("severity"), 0))
    for it in sorted_issues:
        it = it or {}
        sev = (it.get("severity") or "?").upper()
        typ = it.get("type") or "?"
        com = it.get("comment") or ""
        quote = it.get("quote") or ""
        line = f"  [{sev}] {typ}: {com}"
        if quote:
            line += f"  (цитата: «{quote[:120]}»)"
        lines.append(line)
    lines.append("")
    lines.append("Внеси ВСЕ исправления. Сохрани удачные места без изменений.")
    lines.append("Выведи ТОЛЬКО исправленный русский текст сегмента.")
    return "\n".join(lines)


def clean(text):
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[\w]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(
        r"^(Исправленный перевод|Исправленный русский перевод|Финальный перевод"
        r"|Перевод|Результат|Corrected translation)[:：]?\s*\n?",
        "", text, flags=re.IGNORECASE,
    )
    return text.strip()


# ============================================================
# ВАЛИДАЦИЯ ОТВЕТА
# ============================================================

def validate(new, original, source):
    """Простые санитарные проверки, чтобы не подставлять очевидную чушь."""
    if not new or len(new) < 10:
        return False, "пусто/слишком коротко"
    cyr = sum(1 for c in new if 0x0400 <= ord(c) <= 0x04FF)
    if cyr < len(new) * 0.30:
        return False, "мало кириллицы"
    kor = sum(1 for c in new if 0xAC00 <= ord(c) <= 0xD7AF)
    if kor > len(new) * 0.15:
        return False, "много корейского"
    # Не должен сильно сокращать ниже исходного корейского (грубый пропуск).
    if len(new) < len(source) * 0.40:
        return False, "слишком коротко относительно оригинала"
    # Не должен ВДВОЕ перерасти текущий перевод (судья просил починить, а не раздуть).
    if len(new) > len(original) * 2.0 and len(new) > 400:
        return False, "сильно длиннее старого перевода"
    return True, ""


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Этап 9: починка по вердикту судьи через Gemini")
    p.add_argument("--final", default=FINAL_FILE)
    p.add_argument("--judge", default=JUDGE_REPORT_FILE)
    p.add_argument("--glossary", default=GLOSSARY_FILE)
    p.add_argument("--model", default=MODEL_CLOUD)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--mode", choices=["default", "only-bad", "all-review"], default="default",
                   help="default: bad + review с high severity; only-bad: только verdict=bad; "
                        "all-review: bad + любые review")
    p.add_argument("--limit", type=int, default=0,
                   help="Максимум сегментов за один прогон (0 = все)")
    p.add_argument("--ids", default="",
                   help="Конкретные id через запятую (перебивает выборку по вердиктам)")
    p.add_argument("--max-auto", type=int, default=DEFAULT_MAX_AUTO,
                   help=f"Сколько отправок можно без подтверждения (default {DEFAULT_MAX_AUTO})")
    p.add_argument("--yes", action="store_true",
                   help="Не спрашивать подтверждения, отправлять любое количество")
    p.add_argument("--resume", action="store_true",
                   help="Пропускать уже починенные (judge_fix_applied=true) и "
                        "отбитые валидацией (judge_fix_skipped)")
    p.add_argument("--dry-run", action="store_true",
                   help="Только показать кандидатов, без вызовов Gemini")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    # ----- 1. Файлы -----
    if not os.path.exists(args.final):
        print(f"ОШИБКА: нет {args.final}")
        sys.exit(1)
    if not os.path.exists(args.judge):
        print(f"ОШИБКА: нет отчёта судьи {args.judge}. Запусти stage8_judge сперва.")
        sys.exit(1)

    with open(args.final, "r", encoding="utf-8") as f:
        final_data = json.load(f)
    with open(args.judge, "r", encoding="utf-8") as f:
        judge_data = json.load(f)

    final_by_id = {r["id"]: r for r in final_data.get("results", [])}
    judge_by_id = {s["id"]: s for s in judge_data.get("judged", [])}

    glossary = []
    try:
        if os.path.exists(args.glossary):
            with open(args.glossary, "r", encoding="utf-8") as f:
                glossary = json.load(f).get("terms", [])
            print(f"Глоссарий: {len(glossary)} терминов")
    except Exception as e:
        print(f"Глоссарий не загружен ({e}) - работаю без него")

    # ----- 2. Выборка -----
    forced_ids = parse_ids(args.ids)

    targets = []
    for sid in sorted(judge_by_id.keys()):
        if forced_ids:
            if sid not in forced_ids:
                continue
        else:
            judgment = judge_by_id[sid].get("judgment")
            if not should_fix(judgment, args.mode):
                continue
        if sid not in final_by_id:
            print(f"  ! сегмент {sid} есть в judge_report, но нет в 4_final — пропускаю")
            continue
        r = final_by_id[sid]
        if args.resume:
            if r.get("judge_fix_applied"):
                continue
            if r.get("judge_fix_skipped"):
                continue
        targets.append(sid)

    if args.limit > 0:
        targets = targets[:args.limit]

    if not targets:
        print("Нечего чинить.")
        return

    # ----- 3. Сводка кандидатов -----
    print(f"Кандидатов на починку через Gemini ({args.model}): {len(targets)}")
    show_n = min(8, len(targets))
    for sid in targets[:show_n]:
        j = (judge_by_id[sid].get("judgment") or {})
        verdict = j.get("verdict", "?")
        issues = j.get("issues") or []
        n_h = sum(1 for i in issues if (i or {}).get("severity") == "high")
        n_m = sum(1 for i in issues if (i or {}).get("severity") == "medium")
        n_l = sum(1 for i in issues if (i or {}).get("severity") == "low")
        print(f"  #{sid}  verdict={verdict}  H={n_h} M={n_m} L={n_l}  issues={len(issues)}")
    if len(targets) > show_n:
        print(f"  ... и ещё {len(targets) - show_n}")

    # ----- 4. Cost gate -----
    if len(targets) > args.max_auto and not args.yes:
        if sys.stdout.isatty():
            try:
                ans = input(
                    f"\nБудет отправлено {len(targets)} запросов в Gemini "
                    f"(порог --max-auto {args.max_auto}). Продолжить? [y/N]: "
                )
            except EOFError:
                ans = ""
            if ans.strip().lower() not in ("y", "yes", "д", "да"):
                print("Отменено пользователем.")
                return
        else:
            print(f"\nОТКАЗ: будет {len(targets)} запросов > порога --max-auto {args.max_auto}.")
            print("  Запусти вручную с одним из:")
            print("    python stage9_judge_fix.py --resume --yes")
            print(f"    python stage9_judge_fix.py --resume --limit {args.max_auto}")
            print(f"    python stage9_judge_fix.py --resume --max-auto {len(targets)}")
            sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: вызовов Gemini не было, файл не изменён.")
        return

    # ----- 5. Бэкап + клиент -----
    if not args.no_backup:
        backup(args.final)
    client = make_client(read_api_key(args))

    log_lines = [
        f"Stage9: починка по вердикту судьи через Gemini ({args.model})",
        f"Запуск: {datetime.now().isoformat()}",
        f"Кандидатов в этом прогоне: {len(targets)}",
        "=" * 60,
    ]

    # ----- 6. Прогон -----
    fixed = 0
    skipped_validation = 0
    consecutive_empty = 0
    aborted_by_quota = False

    for i, sid in enumerate(targets):
        r = final_by_id[sid]
        src = r.get("source", "")
        current = r.get("final_translation") or r.get("draft_translation", "")
        judgment = judge_by_id[sid].get("judgment") or {}
        issues = judgment.get("issues") or []

        rel = find_relevant_terms(glossary, src, max_terms=MAX_GLOSSARY_TERMS) if glossary else []
        gloss_str = format_glossary_for_prompt(rel) if rel else ""
        gloss_block = (f"ГЛОССАРИЙ (строго):\n{gloss_str}\n\n") if gloss_str else ""

        prompt = JUDGE_FIX_SYSTEM + "\n\n" + build_user_prompt(src, current, issues, gloss_block)

        print(f"[{i+1}/{len(targets)}] #{sid} "
              f"(issues: {len(issues)}, {len(current)} симв) -> Gemini...", flush=True)
        raw = call_gemini(client, args.model, prompt, args.temperature)

        if not raw:
            consecutive_empty += 1
            print(f"  ! пустой ответ Gemini "
                  f"({consecutive_empty}/{ABORT_AFTER_CONSECUTIVE_EMPTY})")
            log_lines.append(f"#{sid}: EMPTY (нет ответа Gemini)")
            if consecutive_empty >= ABORT_AFTER_CONSECUTIVE_EMPTY:
                aborted_by_quota = True
                print(f"\n{ABORT_AFTER_CONSECUTIVE_EMPTY} подряд пустых ответов — "
                      "вероятно исчерпан лимит API или нет сети.")
                print("  Прерываю прогон. Запусти позже с --resume — продолжу с этого сегмента.")
                break
            continue
        consecutive_empty = 0

        new = clean(raw)
        ok, why = validate(new, current, src)
        if not ok:
            skipped_validation += 1
            r["judge_fix_skipped"] = why
            r["judge_fix_attempted_at"] = datetime.now().isoformat()
            print(f"  отбой валидацией ({why}) - оставляю старый перевод")
            log_lines.append(f"#{sid}: SKIP ({why})")
            atomic_save(final_data, args.final)
            if i < len(targets) - 1:
                time.sleep(args.delay)
            continue

        # Применяем правку
        r["pre_judge_fix"] = current
        r["final_translation"] = new
        r["judge_fix_applied"] = True
        r["judge_fix_model"] = args.model
        r["judge_fix_at"] = datetime.now().isoformat()
        r["judge_fix_issues_count"] = len(issues)
        fixed += 1
        preview = new[:90].replace("\n", " ")
        print(f"  OK ({len(new)} симв) -> {preview}{('...' if len(new) > 90 else '')}")
        log_lines.append(f"#{sid}: FIXED (issues={len(issues)}, "
                         f"{len(current)}->{len(new)} симв)")
        atomic_save(final_data, args.final)

        if i < len(targets) - 1:
            time.sleep(args.delay)

    # ----- 7. Финал -----
    log_path = os.path.join(OUTPUT_DIR, "6_judge_fixes.txt")
    log_lines.append("=" * 60)
    log_lines.append(f"Исправлено: {fixed}, отбито валидацией: {skipped_validation}")
    if aborted_by_quota:
        not_processed = len(targets) - fixed - skipped_validation
        log_lines.append(f"Прервано по квоте: не обработано {not_processed} сегментов")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print()
    print("=" * 60)
    print(f"Готово. Исправлено: {fixed}, отбито валидацией: {skipped_validation}")
    if aborted_by_quota:
        not_processed = len(targets) - fixed - skipped_validation
        print(f"! Прерван по {ABORT_AFTER_CONSECUTIVE_EMPTY} пустым ответам подряд "
              f"(вероятно квота). Не обработано: {not_processed}")
        print("  Запусти позже:  python stage9_judge_fix.py --resume")
    print(f"Файл:  {args.final}")
    print(f"Лог:   {log_path}")
    if not aborted_by_quota and fixed:
        print("\nДальше: python export_final.py --format plain  (чтобы пересобрать "
              "final_text.txt с правками)")


if __name__ == "__main__":
    main()
