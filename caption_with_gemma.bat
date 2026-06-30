@echo off
setlocal
REM ===========================================================================
REM caption_with_gemma.bat  --  pull gemma4:26b, then LLM-caption one dataset.
REM Wraps the MarioMaker_llm_captions.py run with everything fixed except the
REM dataset path, which is the only argument.
REM
REM Usage:   caption_with_gemma.bat "C:\path\to\dataset.json"
REM (or just drag the dataset .json onto this .bat in Explorer).
REM Run from the repo root, or double-click it; it cd's to its own folder.
REM ===========================================================================
cd /d "%~dp0"

REM -- The one argument: the input dataset JSON ------------------------------
set DATASET=%~1
if "%DATASET%"=="" (
    echo Usage: %~nx0 "C:\path\to\dataset.json"
    goto end
)
if not exist "%DATASET%" (
    echo Error: dataset not found: %DATASET%
    goto end
)

REM -- Fixed knobs (everything else from the original command) ---------------
set PY=python
set MODEL=gemma4:26b
set OUTPUT=C:\Users\mckeonp\Documents\dataset_captioned.json
set TILESET=mm2_tileset_we.json
set ASCII_DIR=C:\Users\mckeonp\Documents\bcd\ascii_tokens

echo.
echo === [1/2] pulling %MODEL% ===
ollama pull %MODEL%
if errorlevel 1 goto error

echo.
echo === [2/2] captioning %DATASET% ===
%PY% MarioMaker_llm_captions.py --dataset "%DATASET%" --output "%OUTPUT%" --backend ollama --max-tokens 900 --model %MODEL% --tileset %TILESET% --tileset-we %TILESET% --num-captions 5 --grid-format tokens --ascii-output-dir "%ASCII_DIR%"
if errorlevel 1 goto error

echo.
echo === done ===
echo Captions written to "%OUTPUT%".
goto end

:error
echo.
echo *** FAILED at the step above (errorlevel %errorlevel%). Fix that step, then re-run. ***

:end
endlocal
