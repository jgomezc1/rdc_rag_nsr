@echo off
echo ============================================
echo Configurando entorno para NSR-10 + ACI-318
echo ============================================

echo.
echo [1/4] Creando nuevo entorno conda 'nsr_app'...
call conda create -n nsr_app python=3.11 -y

echo.
echo [2/4] Activando entorno...
call conda activate nsr_app

echo.
echo [3/4] Instalando dependencias...
pip install streamlit python-dotenv langchain langchain-community langchain-core langchain-openai chromadb openai pydantic

echo.
echo [4/4] Configuracion completada!
echo.
echo ============================================
echo Para ejecutar la aplicacion:
echo   1. conda activate nsr_app
echo   2. streamlit run app.py
echo ============================================

echo.
echo Ejecutando la aplicacion ahora...
streamlit run app.py

pause
