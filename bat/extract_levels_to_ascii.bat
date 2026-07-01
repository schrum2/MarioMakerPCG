@echo off
REM Usage: extract_levels_to_ascii.bat <name> <count> <output_folder> [extra extract args...]
REM <name>          level name filter; pass "" to skip filtering by name
REM <count>         number of levels to extract
REM <output_folder> folder that will hold bcd/, json/, images/ and ascii/ subfolders
REM [extra]         passed straight to extract_mm2_bcd.py, e.g.
REM                 --tag Speedrun --difficulty Easy Normal
REM                 --likes 1000 --dislikes 300 --exclude-tag Art
cd ..

set NAME=%~1
set COUNT=%~2
set OUTPUT=%~3

if "%COUNT%"=="" (
    echo ERROR: Must provide the number of levels to extract as the second argument.
    exit /b 1
)
if "%OUTPUT%"=="" (
    echo ERROR: Must provide an output folder as the third argument.
    exit /b 1
)

REM Anything past the first three args is forwarded to the extractor as-is.
REM While forwarding, also remember the --tag / --exclude-tag / --difficulty /
REM --likes / --dislikes values so we can note them in extract_info.txt.
shift
shift
shift
set EXTRA=
set TAG_VALUES=
set XTAG_VALUES=
set DIFF_VALUES=
set LIKES_VALUE=
set DISLIKES_VALUE=
set CAPMODE=
:collect_extra
if "%~1"=="" goto after_collect
set EXTRA=%EXTRA% %1
set "CUR=%~1"
if /i "%CUR%"=="--tag" (
    set CAPMODE=tag
) else if /i "%CUR%"=="--exclude-tag" (
    set CAPMODE=xtag
) else if /i "%CUR%"=="--difficulty" (
    set CAPMODE=diff
) else if /i "%CUR%"=="--likes" (
    set CAPMODE=likes
) else if /i "%CUR%"=="--dislikes" (
    set CAPMODE=dislikes
) else if "%CUR:~0,2%"=="--" (
    set CAPMODE=
) else if "%CAPMODE%"=="tag" (
    set "TAG_VALUES=%TAG_VALUES% %~1"
) else if "%CAPMODE%"=="xtag" (
    set "XTAG_VALUES=%XTAG_VALUES% %~1"
) else if "%CAPMODE%"=="diff" (
    set "DIFF_VALUES=%DIFF_VALUES% %~1"
) else if "%CAPMODE%"=="likes" (
    set "LIKES_VALUE=%~1"
) else if "%CAPMODE%"=="dislikes" (
    set "DISLIKES_VALUE=%~1"
)
shift
goto collect_extra
:after_collect
if defined TAG_VALUES set "TAG_VALUES=%TAG_VALUES:~1%"
if defined XTAG_VALUES set "XTAG_VALUES=%XTAG_VALUES:~1%"
if defined DIFF_VALUES set "DIFF_VALUES=%DIFF_VALUES:~1%"

for %%I in ("%OUTPUT%") do set "OUTPUT=%%~fI"

set BCD_DIR=%OUTPUT%\bcd
set JSON_DIR=%OUTPUT%\json
set IMAGES_DIR=%OUTPUT%\images
set ASCII_DIR=%OUTPUT%\ascii
REM Per-level metadata sidecar (level_name/difficulty/gamestyle/theme/tags),
REM keyed by ascii stem. build_dataset_with_ascii.py reads it with --metadata
REM (or auto-detects it since it lives in the ascii folder).
set METADATA=%ASCII_DIR%\metadata.json

REM With a name we cap the matching levels (--name_count); without one we just
REM limit the total. Either way COUNT means "how many levels to extract".
if "%NAME%"=="" (
    set FILTER_ARGS=--limit %COUNT%
) else (
    set FILTER_ARGS=--name "%NAME%" --name_count %COUNT%
)

echo === Step 1: Extracting up to %COUNT% level^(s^) ===
python extract_mm2_bcd.py %FILTER_ARGS% --output_dir "%BCD_DIR%" --skip_3dworld --skip_items --skip_subworld_items%EXTRA%
if %ERRORLEVEL% neq 0 ( echo ERROR: extract_mm2_bcd.py failed. & exit /b 1 )

echo === Step 2: Converting .bcd files to .json and images ===
pushd toost_stuff
python batch_convert.py "%BCD_DIR%" -o "%JSON_DIR%" --images-output "%IMAGES_DIR%"
if %ERRORLEVEL% neq 0 ( popd & echo ERROR: batch_convert.py failed. & exit /b 1 )
popd

echo === Step 3: Converting .json files to ASCII ===
python mm2_json_to_ascii.py "%JSON_DIR%" "%ASCII_DIR%" --metadata_output "%METADATA%"
if %ERRORLEVEL% neq 0 ( echo ERROR: mm2_json_to_ascii.py failed. & exit /b 1 )

REM Record the name filter and the level count (one .txt per level) so
REM summarize_dataset.py can pick them up later when it summarizes the dataset.
set LEVEL_COUNT=0
for %%F in ("%ASCII_DIR%\*.txt") do set /a LEVEL_COUNT+=1
(
echo name=%NAME%
echo requested_count=%COUNT%
echo levels_extracted=%LEVEL_COUNT%
if defined TAG_VALUES echo tags=%TAG_VALUES%
if defined XTAG_VALUES echo excluded_tags=%XTAG_VALUES%
if defined DIFF_VALUES echo difficulty=%DIFF_VALUES%
if defined LIKES_VALUE echo min_likes=%LIKES_VALUE%
if defined DISLIKES_VALUE echo max_dislikes=%DISLIKES_VALUE%
)>"%OUTPUT%\extract_info.txt"

echo === Done! ===
echo   BCDs:     %BCD_DIR%
echo   JSON:     %JSON_DIR%
echo   Images:   %IMAGES_DIR%
echo   ASCII:    %ASCII_DIR%
echo   Metadata: %METADATA%
echo.
echo Build the dataset with the metadata folded in, e.g.:
echo   python build_dataset_with_ascii.py --input_file "%ASCII_DIR%" --output dataset.json --tileset mm2_tileset_we.json --metadata "%METADATA%"
