@echo off
cd /d "%~dp0"   REM Change working directory to the folder of this .bat file

REM Set the script path, the directory for the images and the model to be used
set "SCRIPT=..\src\text_detect.py"
set "MODEL=..\models\some_model_path"

REM Run the script on text images
python "%SCRIPT%" "<path1>" "%MODEL%"
python "%SCRIPT%" "<path2>" "%MODEL%"
python "%SCRIPT%" "<path3>" "%MODEL%"
python "%SCRIPT%" "<path4>" "%MODEL%"
python "%SCRIPT%" "<path5>" "%MODEL%"