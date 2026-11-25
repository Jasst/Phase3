@echo off
setlocal
pushd "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make-archive.ps1" %*
set "rc=%ERRORLEVEL%"
popd
pause
endlocal & exit /b %rc%
