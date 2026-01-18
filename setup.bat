@echo off
echo ================================================
echo Setting up YT Bulk Subtitles Downloader
echo ================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer and try again.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    echo Make sure you have Python's venv module available.
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Warning: Some dependencies may have failed to install.
    echo Check the error messages above.
    echo You can try running the setup script again.
)
echo.

echo ================================================
echo Setup complete!
echo ================================================
echo.
echo To run YT Bulk Subtitles Downloader, use start.bat
echo.
pause