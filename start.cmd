@echo off

SETLOCAL

# Memory error prevention
set PYTORCH_ALLOC_CONF=expandable_segments:True

echo Parameters:
echo Start: %START%
echo End: %END%
echo Root directory: %ROOT_DIR%
echo Memory history directory: %MEM_HISTORY_DIR%
echo Train-test datasets proportion: %PROPORTION%

IF "%START%"=="" (
	set START=0
)

SET exec_path=%~dp0
SET exec_path=%exec_path:~0,-1%
SET exec_path=%exec_path%\src

set i=%START%

:loopstart

echo Starting action %i%
SET ACTION_ID=%i%
SET MEM_HISTORY_OUT=%MEM_HISTORY_DIR%\%i%.pkl
py %exec_path%
set el=%ERRORLEVEL%
echo Action %i% ended

if %i% EQU %END% goto loopend
if %el% NEQ 0 goto loopend

set /a i=i+1
goto loopstart

:loopend

ENDLOCAL
