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
  echo [1/11] split - already done, skip
) else (
  echo [1/11] stage1_split - split...
  %PY% stage1_split.py >> "%LOG%" 2>&1
  if errorlevel 1 goto fail
)

rem ---- 2: glossary ----
if exist "output\2_glossary.json" (
  echo [2/11] glossary - already done, skip
) else (
  echo [2/11] stage2_glossary - glossary qwen...
  %PY% stage2_glossary.py >> "%LOG%" 2>&1
  if errorlevel 1 goto fail
)

rem ---- 3: glossary cross-check (stage3 creates 2_glossary.qwen.json) ----
if exist "output\2_glossary.qwen.json" (
  echo [3/11] glossary check - already done, skip
) else (
  echo [3/11] stage3_glossary_check - glossary check aya...
  %PY% stage3_glossary_check.py >> "%LOG%" 2>&1
  if errorlevel 1 goto fail
)

echo [4/11] stage4_translate - translate qwen, resume...
%PY% stage4_translate.py --resume >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [5/11] stage5_translate_check - translation check aya, resume...
%PY% stage5_translate_check.py --resume >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [6/11] gemini_fallback - fix what local models could not...
%PY% gemini_fallback.py --include-check-failed >> "%LOG%" 2>&1
if errorlevel 1 echo    ! gemini_fallback error, continuing - see log

echo [7/11] stage6_proofread - literary proofread gemma, resume...
%PY% stage6_proofread.py --resume --export-txt >> "%LOG%" 2>&1
if errorlevel 1 goto fail

echo [8/11] gemini_fallback --mode polish - polish leftover fallbacks...
%PY% gemini_fallback.py --mode polish >> "%LOG%" 2>&1
if errorlevel 1 echo    ! gemini_fallback polish error, continuing - see log

echo [9/11] stage7_qa - mechanical QA...
%PY% stage7_qa.py >> "%LOG%" 2>&1
if errorlevel 1 echo    ! stage7_qa error, continuing

echo [10/11] stage8_judge - final judge Magistral, resume...
%PY% stage8_judge.py --resume >> "%LOG%" 2>&1
if errorlevel 1 echo    ! stage8_judge error, continuing

echo [11/11] export_final - clean text...
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
