@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ============================================================
rem  RESUME-FRIENDLY runner.
rem  - If interrupted, just run this .bat again: finished early stages
rem    are skipped, translate/check/proofread/judge continue from where
rem    they stopped (per-segment progress is saved to disk).
rem  - For a NEW chapter: first MOVE or CLEAR the output\ folder,
rem    otherwise old segments/glossary/translation will be reused.
rem ============================================================

set "PY=python"
rem If 'python' is not found, change to:  set "PY=py -3"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "LOG=output\pipeline.log"

echo. >> "%LOG%" 2>nul
echo ==== Pipeline run: %DATE% %TIME% ==== >> "%LOG%" 2>nul

echo ============================================================
echo  AI Light Novel Translator - full run, resume-friendly
echo  Full log: %LOG%
echo ============================================================
echo.

rem ---- Preflight: is Ollama alive ----
%PY% -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=5)" >nul 2>&1
if errorlevel 1 (
  echo Ollama is not responding on http://localhost:11434 - run 'ollama serve' and retry.
  goto end
)

rem ---- 1: split ----
if exist "output\1_segments.json" (
  echo [1/12] split - already done, skip
) else (
  echo [1/12] stage1_split - split...
  %PY% stage1_split.py >> "%LOG%" 2>&1
  if errorlevel 1 goto fail
)

rem ---- 2: glossary ----
if exist "output\2_glossary.json" (
  echo [2/12] glossary - already done, skip
) else (
  echo [2/12] stage2_glossary - glossary qwen...
  %PY% stage2_glossary.py >> "%LOG%" 2>&1
  if errorlevel 1 goto fail
)

rem ---- 3: glossary cross-check (stage3 creates 2_glossary.qwen.json) ----
if exist "output\2_glossary.qwen.json" (
  echo [3/12] glossary check - already done, skip
) else (
  echo [3/12] stage3_glossary_check - glossary check aya...
  %PY% stage3_glossary_check.py >> "%LOG%" 2>&1
  if errorlevel 1 goto fail
)

rem ---- 4: gemini arbiter for glossary disagreements ----
rem    Idempotent: if disagreements file is missing or empty, just no-ops.
rem    Skips gracefully without an API key (errorlevel != 0, but we continue).
echo [4/12] gemini_fallback --mode glossary - arbitrate qwen/aya glossary disagreements...
%PY% gemini_fallback.py --mode glossary >> "%LOG%" 2>&1
if errorlevel 1 echo    ! gemini_fallback glossary error, continuing - see log

echo [5/12] stage4_translate - translate qwen, resume...
%PY% stage4_translate.py --resume >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [6/12] stage5_translate_check - translation check aya, resume...
%PY% stage5_translate_check.py --resume >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [7/12] gemini_fallback - fix what local models could not...
%PY% gemini_fallback.py --include-check-failed >> "%LOG%" 2>&1
if errorlevel 1 echo    ! gemini_fallback error, continuing - see log

echo [8/12] stage6_proofread - literary proofread gemma, resume...
%PY% stage6_proofread.py --resume --export-txt >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [9/12] gemini_fallback --mode polish - polish leftover fallbacks...
%PY% gemini_fallback.py --mode polish >> "%LOG%" 2>&1
if errorlevel 1 echo    ! gemini_fallback polish error, continuing - see log

echo [10/12] stage7_qa - mechanical QA...
%PY% stage7_qa.py >> "%LOG%" 2>&1
if errorlevel 1 echo    ! stage7_qa error, continuing

echo [11/12] stage8_judge - final judge Magistral, resume...
%PY% stage8_judge.py --resume >> "%LOG%" 2>&1
if errorlevel 1 echo    ! stage8_judge error, continuing

echo [12/12] export_final - clean text...
%PY% export_final.py --format plain >> "%LOG%" 2>&1

echo.
echo ============================================================
echo  DONE %TIME%. Results in output\:
echo    4_final.txt / final_text.txt  - final translation
echo    5_qa_report.txt               - mechanical QA
echo    6_judge_report.txt            - judge verdict, bring this back
echo  Full log: %LOG%
echo  NOTE: for a NEW chapter, clear or move the output\ folder first.
echo ============================================================
goto end

:fail
echo.
echo !!! CRITICAL ERROR at the stage above - run stopped.
echo     You can simply run this .bat again - it resumes from where it stopped.
echo     Log tail:
powershell -NoProfile -Command "if (Test-Path '%LOG%') { Get-Content -Tail 25 -Encoding utf8 '%LOG%' }" 2>nul
echo     Common causes: Ollama not running; wrong model tag in config - check 'ollama list'.

:end
echo.
pause
endlocal
