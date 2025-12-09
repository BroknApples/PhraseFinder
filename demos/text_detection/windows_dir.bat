@echo off
cd /d "%~dp0"   REM Change working directory to the folder of this .bat file

REM Set the script path, the directory for the images and the model to be used
set "SCRIPT=..\..\src\text_detect.py"
set "IMG_DIR=..\..\some_image_dirpath"
set "MODEL=..\..\models\some_model_path"

REM Run the script on each file
for %%f in ("%IMG_DIR%\*") do (
  python "%SCRIPT%" "%%~f" "%MODEL%"
)