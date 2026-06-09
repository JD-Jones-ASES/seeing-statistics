@echo off
REM ============================================================
REM  Double-click this file to open the statistics course.
REM  It launches JupyterLab from the project's private toolbox
REM  and opens your web browser. Close the black window to stop.
REM ============================================================
pushd "%~dp0"
"%~dp0..\.venv\Scripts\jupyter-lab.exe"
popd
pause
