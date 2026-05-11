param(
    [switch]$InstallDeps,
    [switch]$OneFile,
    [string]$Name = "EchoSegUI"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Step "Checking Python"
$Python = (Get-Command python -ErrorAction Stop).Source
python --version

if ($InstallDeps) {
    Step "Installing project dependencies"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
}

Step "Checking PyInstaller"
python -m PyInstaller --version | Out-Host

Step "Cleaning old build output"
if (Test-Path "build") {
    Remove-Item -LiteralPath "build" -Recurse -Force
}
if (Test-Path "dist\$Name") {
    Remove-Item -LiteralPath "dist\$Name" -Recurse -Force
}
if (Test-Path "dist\$Name.exe") {
    Remove-Item -LiteralPath "dist\$Name.exe" -Force
}

$Mode = if ($OneFile) { "--onefile" } else { "--onedir" }
$Separator = ";"

$Args = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    $Mode,
    "--windowed",
    "--name", $Name,
    "--collect-all", "torch",
    "--collect-all", "torchvision",
    "--collect-all", "cv2",
    "--hidden-import", "PIL._tkinter_finder",
    "--hidden-import", "skimage.draw",
    "--hidden-import", "scipy.signal",
    "--add-data", "utils${Separator}utils",
    "--add-data", "main.py${Separator}.",
    "app.py"
)

Step "Building $Name.exe"
python @Args

Step "Build finished"
if ($OneFile) {
    Write-Host "Executable: $ProjectRoot\dist\$Name.exe" -ForegroundColor Green
} else {
    Write-Host "Executable: $ProjectRoot\dist\$Name\$Name.exe" -ForegroundColor Green
}
Write-Host "Keep data, weights, and output folders next to the exe or select them from the UI."
