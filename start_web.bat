@echo off
echo ==========================================
echo   TINNAM AI - PRIVATE WEB ACCESS
echo ==========================================
echo.

:: Start Streamlit in background
echo [1/2] Starting TinNam AI...
start /B python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true

:: Wait for Streamlit to start
timeout /t 5 /nobreak > nul

:: Start Cloudflare Tunnel
echo [2/2] Creating private tunnel...
echo.
echo ============================================
echo   Khi thay URL xuat hien, copy va mo tren
echo   bat ky may tinh/dien thoai nao.
echo   Mat khau: 1991
echo ============================================
echo.
cloudflared.exe tunnel --url http://localhost:8501
pause
