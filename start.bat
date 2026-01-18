@echo off
echo ================================================
echo Starting YT Bulk Subtitles Downloader
echo ================================================
echo.

REM Check if the virtual environment exists
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found.
    echo Please run setup.bat first to set up the environment.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Run the downloader
python ytbsd.py

REM This part runs if the program exits or crashes
echo.
echo ================================================
echo Program has stopped. Check the output above for any errors.
echo ================================================
echo.
pause