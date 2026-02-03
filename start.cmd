@echo off

SETLOCAL

SET exec_path=%~dp0
SET exec_path=%exec_path:~0,-1%
SET exec_path=%exec_path%\src

set i=0

:loopstart

echo Starting action %i%
SET ACTION_ID=%i%
SET MEM_HISTORY_OUT=%MEM_HISTORY_DIR%\%i%.pkl
py %exec_path%
set el=%ERRORLEVEL%
echo Action %i% ended

if %el% NEQ 0 goto loopend

set /a i=i+1
if %i% NEQ 193 goto loopstart

:loopend

ENDLOCAL
