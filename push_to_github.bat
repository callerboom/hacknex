@echo off
title Push to GitHub - callerboom/hacknex
cd /d "%~dp0"
echo ========================================================
echo Pushing AI LOCAL to https://github.com/callerboom/hacknex.git
echo ========================================================
echo.
echo If prompted, sign in to your GitHub account (callerboom) in the browser.
echo.
git push -u origin main
echo.
pause
