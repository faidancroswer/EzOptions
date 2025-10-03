@echo off
title EzOptions Trading System
echo.
echo ========================================
echo   EzOptions Trading System Launcher
echo ========================================
echo.
echo [1] Iniciar Dashboard EzOptions
echo [2] Iniciar Sistema de Trading
echo [3] Iniciar Ambos os Sistemas
echo [4] Ver Status dos Sistemas
echo [5] Sair
echo.
set /p opcao="Escolha uma opcao (1-5): "

if "%opcao%"=="1" (
    echo.
    echo Iniciando Dashboard EzOptions...
    start http://localhost:8501
    streamlit run ezoptions.py --server.port 8501
) else if "%opcao%"=="2" (
    echo.
    echo Iniciando Sistema de Trading...
    python main_trading_system.py
) else if "%opcao%"=="3" (
    echo.
    echo Iniciando Ambos os Sistemas...
    start http://localhost:8501
    start cmd /k "python main_trading_system.py"
    streamlit run ezoptions.py --server.port 8501
) else if "%opcao%"=="4" (
    echo.
    echo Verificando status dos sistemas...
    echo.
    echo Dashboard: http://localhost:8501
    echo.
    python trading_system.py
    pause
) else if "%opcao%"=="5" (
    echo Saindo...
    exit
) else (
    echo Opcao invalida! Tente novamente.
    pause
    call "%~f0"
)