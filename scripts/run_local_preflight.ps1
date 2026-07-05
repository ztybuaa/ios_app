$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
chcp 65001 | Out-Null

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Project virtual environment is missing: $VenvPython"
}

$env:npm_config_cache = Join-Path $Root "build\npm-cache"
$env:PYTHONUTF8 = "1"

Push-Location $Root
try {
    Write-Host "== iOS project structure =="
    & $VenvPython "scripts\validate_ios_project.py"

    Write-Host ""
    Write-Host "== Node dependencies =="
    npm install

    Write-Host ""
    Write-Host "== Semantic image eval fixtures =="
    & $VenvPython "scripts\prepare_semantic_eval_dataset.py"

    Write-Host ""
    Write-Host "== MobileCLIP semantic image retrieval =="
    npm run eval:semantic-images
}
finally {
    Pop-Location
}
