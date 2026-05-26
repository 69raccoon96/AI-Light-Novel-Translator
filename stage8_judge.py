"""
Этап 8: Финальный СУДЬЯ качества перевода (LLM-as-judge).
По умолчанию - Mistral Magistral Small 24B (magistral:24b), рассуждающая модель.

Что делает:
  - Для каждого сегмента сверяет финальный русский перевод (4_final.json) с
    корейским оригиналом и глоссарием.
  - Выставляет баллы (accuracy / fluency / terminology / completeness, 1-5) и
    перечисляет КОНКРЕТНЫЕ проблемы (тип, серьёзность, цитата, комментарий).
  - НИЧЕГО НЕ МЕНЯЕТ в переводе - только оценивает.
  - Пишет 6_judge_report.json (полные данные) и 6_judge_report.txt
    (человекочитаемый список проблем, отсортированный по серьёзности).

Magistral даёт <think>...</think> - это нормально, скрипт его срезает и достаёт
итоговый JSON.

Запуск:
  python stage8_judge.py                 # судить весь 4_final.json
  python stage8_judge.py --resume        # продолжить, пропуская уже оценённые
  python stage8_judge.py --start 1 --end 10
  python stage8_judge.py --dry-run       # сколько сегментов будет оценено (без вызовов)
  python stage8_judge.py --only-flagged  # в txt только проблемные (verdict != ok)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests

from config import *

try:
    from stage4_translate import find_relevant_terms, format_glossary_for_prompt
except Exception:
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


MAX_GLOSSARY_TERMS = 50

JUDGE_SYSTEM = (
    "Ты - строгий билингвальный судья качества художественного перевода с "
    "корейского на русский (ранобэ/веб-новеллы). Тебе дают корейский оригинал и "
    "его финальный русский перевод. Твоя задача - ОЦЕНИТЬ перевод и перечислить "
    "конкретные недостатки. Ты НИЧЕГО НЕ ПЕРЕПИСЫВАЕШЬ - только судишь.\n\n"
    "Проверяй:\n"
    "- accuracy (точность смысла): искажения, неверные омонимы/ханча, перепутанные "
    "числа, неправильно понятые слова;\n"
    "- completeness (полнота): пропуски смысла И добавленные/выдуманные реплики или "
    "детали, которых нет в оригинале (галлюцинации); непереведённые куски/остатки корейского;\n"
    "- terminology: соответствие глоссарию (он в сообщении); фонетические кальки "
    "концепт-терминов вместо смысла (напр. Гуин Цзюймэй вместо меридианы Девяти Инь);\n"
    "- fluency (естественность): кальки с корейского, канцелярит, корявый синтаксис, "
    "неестественные диалоги, сбитый тон/регистр.\n\n"
    "Баллы 1-5 (5 - отлично, 1 - плохо). verdict: \"ok\" (можно публиковать), "
    "\"review\" (есть что поправить), \"bad\" (серьёзные проблемы).\n"
    "Для каждой проблемы укажи severity: \"high\" (искажает смысл/галлюцинация/"
    "непереведено), \"medium\" (терминология/заметная корявость), \"low\" (мелочь).\n"
    "type из набора: mistranslation, omission, addition, terminology, untranslated, "
    "fluency, tone, number, name.\n"
    "В поле quote дай короткую цитату из русского перевода (или из оригинала), к "
    "которой относится замечание.\n\n"
    "Отвечай СТРОГО одним JSON-объектом, без пояснений вне JSON (рассуждать можешь "
    "внутри <think>, но финальный ответ - только JSON):\n"
    "{\"scores\":{\"accuracy\":N,\"fluency\":N,\"terminology\":N,\"completeness\":N},"
    "\"issues\":[{\"severity\":\"high|medium|low\",\"type\":\"...\",\"quote\":\"...\","
    "\"comment\":\"...\"}],\"verdict\":\"ok|review|bad\"}\n"
    "Если перевод хорош - верни пустой список issues и verdict \"ok\"."
)

SEV_RANK = {"high": 3, "medium": 2, "low": 1}
VERDICT_RANK = {"bad": 2, "review": 1, "ok": 0, "parse_error": 3}


def call_ollama(system, user, model, num_ctx=JUDGE_NUM_CTX,
                temperature=JUDGE_TEMPERATURE, num_predict=JUDGE_NUM_PREDICT,
                keep_alive="30m", max_retries=3):
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
            r = requests.post(url, json=payload, timeout=1800)
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
            print("  Подсказка: 404 = модель не найдена. Сверьте MODEL_JUDGE с `ollama list`.")
            return ""
        except Exception as e:
            last = str(e)
            print(f"\n  ! Ошибка: {e} ({a}/{max_retries})")
            time.sleep(5)
    print(f"  ! Не удалось за {max_retries} попыток ({last})")
    return ""


def parse_judgment(resp):
    """Срезает <think>, достаёт итоговый JSON-объект. Возвращает dict либо None."""
    if not resp:
        return None
    resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
    resp = re.sub(r"<think>.*", "", resp, flags=re.DOTALL)   # незакрытый think
    resp = re.sub(r"```\w*\n?", "", resp).replace("```", "")
    # берём ПОСЛЕДНИЙ {...} - итоговый ответ обычно в конце
    matches = list(re.finditer(r"\{[\s\S]*\}", resp))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    # запасной разбор по балансу скобок
    start = resp.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(resp)):
            if resp[i] == "{":
                depth += 1
            elif resp[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(resp[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def normalize_judgment(j):
    """Приводит ответ модели к единой структуре."""
    if not isinstance(j, dict):
        return None
    scores = j.get("scores") or {}
    if not isinstance(scores, dict):
        scores = {}

    def sc(k):
        v = scores.get(k)
        try:
            return max(1, min(5, int(round(float(v)))))
        except (TypeError, ValueError):
            return None

    issues = []
    for it in (j.get("issues") or []):
        if not isinstance(it, dict):
            continue
        sev = str(it.get("severity", "")).strip().lower()
        if sev not in SEV_RANK:
            sev = "medium"
        issues.append({
            "severity": sev,
            "type": str(it.get("type", "")).strip() or "other",
            "quote": str(it.get("quote", "")).strip()[:200],
            "comment": str(it.get("comment", "")).strip()[:400],
        })
    verdict = str(j.get("verdict", "")).strip().lower()
    if verdict not in ("ok", "review", "bad"):
        verdict = "bad" if any(i["severity"] == "high" for i in issues) else (
            "review" if issues else "ok")
    return {
        "scores": {k: sc(k) for k in ("accuracy", "fluency", "terminology", "completeness")},
        "issues": issues,
        "verdict": verdict,
    }


def judge_segment(source, translation, glossary_terms, model, temperature):
    rel = find_relevant_terms(glossary_terms, source, max_terms=MAX_GLOSSARY_TERMS) if glossary_terms else []
    gloss = format_glossary_for_prompt(rel) if rel else ""
    gloss_block = (f"ГЛОССАРИЙ (имена/термины должны быть так):\n{gloss}\n\n") if gloss else ""
    user = (f"{gloss_block}КОРЕЙСКИЙ ОРИГИНАЛ:\n---\n{source}\n---\n\n"
            f"ФИНАЛЬНЫЙ РУССКИЙ ПЕРЕВОД:\n---\n{translation}\n---\n\n"
            "Оцени перевод и верни JSON-вердикт по схеме из инструкции.")
    raw = call_ollama(JUDGE_SYSTEM, user, model=model, temperature=temperature)
    parsed = normalize_judgment(parse_judgment(raw)) if raw else None
    return parsed, raw


def atomic_save(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def avg(nums):
    nums = [n for n in nums if isinstance(n, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def write_txt_report(report, txt_path, only_flagged=False):
    judged = report["judged"]
    meta = report["metadata"]
    lines = []
    lines.append("ОТЧЁТ СУДЬИ ПЕРЕВОДА")
    lines.append("=" * 60)
    lines.append(f"Модель судьи: {meta.get('judge_model')}")
    lines.append(f"Источник: {meta.get('source_file')}")
    _tot = meta.get('total')
    lines.append(f"Оценено сегментов: {meta.get('judged')}" + (f"/{_tot}" if _tot else ""))
    a = meta.get("avg_scores", {})
    lines.append(f"Средние баллы: accuracy {a.get('accuracy')} | fluency {a.get('fluency')} "
                 f"| terminology {a.get('terminology')} | completeness {a.get('completeness')}")
    vc = meta.get("verdict_counts", {})
    lines.append(f"Вердикты: ok {vc.get('ok',0)} | review {vc.get('review',0)} "
                 f"| bad {vc.get('bad',0)} | parse_error {vc.get('parse_error',0)}")
    sc = meta.get("severity_counts", {})
    lines.append(f"Проблемы: high {sc.get('high',0)} | medium {sc.get('medium',0)} | low {sc.get('low',0)}")
    tc = meta.get("type_counts", {})
    if tc:
        lines.append("Типы проблем: " + ", ".join(f"{k} {v}" for k, v in
                     sorted(tc.items(), key=lambda x: -x[1])))
    lines.append("")
    lines.append("-" * 60)
    lines.append("СЕГМЕНТЫ (по убыванию серьёзности)")
    lines.append("-" * 60)

    def seg_sort_key(s):
        j = s.get("judgment") or {}
        issues = j.get("issues", [])
        high = sum(1 for i in issues if i["severity"] == "high")
        med = sum(1 for i in issues if i["severity"] == "medium")
        return (VERDICT_RANK.get(j.get("verdict", "ok"), 0), high, med, len(issues))

    for s in sorted(judged, key=seg_sort_key, reverse=True):
        j = s.get("judgment") or {}
        verdict = j.get("verdict", "?")
        issues = j.get("issues", [])
        if only_flagged and verdict == "ok" and not issues:
            continue
        sc2 = j.get("scores", {})
        lines.append("")
        lines.append(f"#{s['id']}  verdict={verdict}  "
                     f"acc={sc2.get('accuracy')} flu={sc2.get('fluency')} "
                     f"term={sc2.get('terminology')} comp={sc2.get('completeness')}")
        if s.get("parse_error"):
            lines.append("  ! не удалось распарсить ответ судьи (см. raw в json)")
        for i in sorted(issues, key=lambda x: -SEV_RANK.get(x["severity"], 0)):
            lines.append(f"  [{i['severity'].upper()}] {i['type']}: {i['comment']}")
            if i.get("quote"):
                lines.append(f"      цитата: {i['quote']}")
        prev = (s.get("preview") or "").replace("\n", " ")
        lines.append(f"  превью перевода: {prev}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_metadata(model, source_file, judged, total=None):
    sev_counts = {"high": 0, "medium": 0, "low": 0}
    type_counts = {}
    verdict_counts = {"ok": 0, "review": 0, "bad": 0, "parse_error": 0}
    acc, flu, term, comp = [], [], [], []
    for s in judged:
        j = s.get("judgment")
        if s.get("parse_error") or not j:
            verdict_counts["parse_error"] += 1
            continue
        verdict_counts[j.get("verdict", "ok")] = verdict_counts.get(j.get("verdict", "ok"), 0) + 1
        for i in j.get("issues", []):
            sev_counts[i["severity"]] = sev_counts.get(i["severity"], 0) + 1
            type_counts[i["type"]] = type_counts.get(i["type"], 0) + 1
        sc = j.get("scores", {})
        acc.append(sc.get("accuracy")); flu.append(sc.get("fluency"))
        term.append(sc.get("terminology")); comp.append(sc.get("completeness"))
    return {
        "judge_model": model,
        "source_file": source_file,
        "judged": len(judged),
        "total": total,
        "avg_scores": {"accuracy": avg(acc), "fluency": avg(flu),
                       "terminology": avg(term), "completeness": avg(comp)},
        "verdict_counts": verdict_counts,
        "severity_counts": sev_counts,
        "type_counts": type_counts,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    p = argparse.ArgumentParser(description="Этап 8: финальный судья перевода (Magistral)")
    p.add_argument("--source", choices=["final", "translated"], default="final")
    p.add_argument("--final", default=FINAL_FILE)
    p.add_argument("--translated", default=TRANSLATED_FILE)
    p.add_argument("--glossary", default=GLOSSARY_FILE)
    p.add_argument("--output", default=JUDGE_REPORT_FILE)
    p.add_argument("--model", default=MODEL_JUDGE)
    p.add_argument("--temperature", type=float, default=JUDGE_TEMPERATURE)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only-flagged", action="store_true",
                   help="В txt-отчёт писать только проблемные (verdict != ok)")
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # --- источник перевода ---
    if args.source == "final" and os.path.exists(args.final):
        path = args.final
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("results", [])
        text_key = "final_translation"
        alt_key = "draft_translation"
    elif os.path.exists(args.translated):
        path = args.translated
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("translations", [])
        text_key = "translation"
        alt_key = "translation"
    else:
        print("ОШИБКА: не найден ни 4_final.json, ни 3_translated.json")
        sys.exit(1)

    if not items:
        print(f"ОШИБКА: в {path} нет сегментов")
        sys.exit(1)

    glossary = []
    try:
        if os.path.exists(args.glossary):
            with open(args.glossary, "r", encoding="utf-8") as f:
                glossary = json.load(f).get("terms", [])
            print(f"Глоссарий: {len(glossary)} терминов")
    except Exception as e:
        print(f"Глоссарий не загружен ({e}) - работаю без него")

    # --- что уже оценено (resume) ---
    prev_judged = {}
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                old = json.load(f)
            for s in old.get("judged", []):
                prev_judged[s["id"]] = s
            print(f"Восстановлено оценок: {len(prev_judged)}")
        except Exception:
            pass

    end = args.end if args.end > 0 else max(t["id"] for t in items)
    todo = []
    for t in sorted(items, key=lambda x: x["id"]):
        if not (args.start <= t["id"] <= end):
            continue
        if args.resume and t["id"] in prev_judged:
            continue
        todo.append(t)
    if args.limit > 0:
        todo = todo[:args.limit]

    if not todo:
        print("Нечего судить.")
        return

    print(f"Судья: {args.model} | сегментов к оценке: {len(todo)} | источник: {path}")
    if args.dry_run:
        print("--dry-run: вызовов модели не было.")
        return
    print("-" * 60)

    judged_map = dict(prev_judged)
    start_t = time.time()
    done = 0
    for idx, t in enumerate(todo):
        sid = t["id"]
        src = t.get("source", "")
        tr = t.get(text_key) or t.get(alt_key) or ""
        if done > 0:
            avg_t = (time.time() - start_t) / done
            eta = (datetime.now() + timedelta(seconds=(len(todo) - idx) * avg_t)).strftime("%H:%M %d.%m")
            es = f" | ETA {eta}"
        else:
            es = ""
        print(f"[{idx+1}/{len(todo)}] #{sid} ({len(tr)} симв){es}", flush=True)

        if not tr or len(tr.strip()) < 5:
            judged_map[sid] = {"id": sid, "preview": "", "judgment": None,
                               "parse_error": False, "note": "пустой перевод"}
        else:
            t0 = time.time()
            judgment, raw = judge_segment(src, tr, glossary, args.model, args.temperature)
            rec = {"id": sid, "preview": tr[:160], "judgment": judgment,
                   "parse_error": judgment is None}
            if judgment is None:
                rec["raw"] = (raw or "")[:1500]
                print(f"  ! не распарсил вердикт ({time.time()-t0:.0f}с)")
            else:
                v = judgment.get("verdict")
                ni = len(judgment.get("issues", []))
                print(f"  {v} | проблем: {ni} ({time.time()-t0:.0f}с)")
            judged_map[sid] = rec
        done += 1

        judged_list = sorted(judged_map.values(), key=lambda x: x["id"])
        report = {"metadata": build_metadata(args.model, path, judged_list, total=len(items)),
                  "judged": judged_list}
        atomic_save(report, args.output)
        if idx < len(todo) - 1:
            time.sleep(args.delay)

    judged_list = sorted(judged_map.values(), key=lambda x: x["id"])
    report = {"metadata": build_metadata(args.model, path, judged_list, total=len(items)),
              "judged": judged_list}
    atomic_save(report, args.output)
    txt_path = args.output.replace(".json", ".txt")
    write_txt_report(report, txt_path, only_flagged=args.only_flagged)

    m = report["metadata"]
    print("\n" + "=" * 60)
    print(f"Готово. Оценено: {m['judged']}")
    print(f"Средние баллы: {m['avg_scores']}")
    print(f"Вердикты: {m['verdict_counts']}")
    print(f"Проблемы по серьёзности: {m['severity_counts']}")
    print(f"Отчёт JSON: {args.output}")
    print(f"Отчёт TXT:  {txt_path}")
    print("\nПринеси список из 6_judge_report.txt - разберём рекомендации.")


if __name__ == "__main__":
    main()
