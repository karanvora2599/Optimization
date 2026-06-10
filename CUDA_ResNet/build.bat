@echo off
:: build.bat — One-shot build script for CUDA_ResNet on Windows.
:: Run from the CUDA_ResNet directory in a plain Command Prompt or PowerShell.
:: Does NOT require a "Developer Command Prompt" — this script locates MSVC itself.

setlocal EnableDelayedExpansion

:: ── Locate MSVC (installed by VS Build Tools or full Visual Studio) ───────────
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found.
    echo Install Visual Studio Build Tools first:
    echo   winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --norestart"
    exit /b 1
)

for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo ERROR: No MSVC installation found. Install Visual Studio Build Tools.
    exit /b 1
)

:: Initialise MSVC environment (adds cl.exe, lib.exe, link.exe to PATH)
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
echo MSVC found at: %VS_PATH%

:: ── Verify NVCC is on PATH ────────────────────────────────────────────────────
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: nvcc not found on PATH.
    echo Add the CUDA bin directory to your PATH:
    echo   set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%%PATH%%
    exit /b 1
)
for /f "tokens=*" %%v in ('nvcc --version ^| findstr "release"') do echo NVCC: %%v

:: ── Configure and build ───────────────────────────────────────────────────────
if not exist build mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed.
    exit /b 1
)

cmake --build . --config Release --parallel
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b 1
)

echo.
echo ====================================================
echo  Build successful: build\Release\CUDA_ResNet.exe
echo ====================================================
echo.
echo Run with:
echo   build\Release\CUDA_ResNet.exe "C:\Users\karan\Documents\Optimization Techniques\CUDA_ResNet\data"
