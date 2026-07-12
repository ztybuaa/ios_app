$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
chcp 65001 | Out-Null

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Project virtual environment is missing: $VenvPython"
}

$env:PYTHONUTF8 = "1"

Push-Location $Root
try {
    Write-Host "== iOS project structure =="
    & $VenvPython "scripts\validate_ios_project.py"

    Write-Host ""
    Write-Host "== Resource query normalization =="
    & $VenvPython "scripts\validate_resource_query_normalization.py"

    Write-Host ""
    Write-Host "== Non-photo retrieval ranking =="
    & $VenvPython "scripts\eval_non_photo_retrieval.py"

    Write-Host ""
    Write-Host "== Semantic indexing performance policy =="
    & $VenvPython "scripts\validate_semantic_search_performance.py"

    Write-Host ""
    Write-Host "== Semantic image eval fixtures =="
    & $VenvPython "scripts\prepare_semantic_eval_dataset.py"

    Write-Host ""
    Write-Host "== Chinese-CLIP RN50 evaluation dependencies =="
    & $VenvPython -m pip install -r "scripts\requirements\chinese_clip_eval.txt"

    Write-Host ""
    Write-Host "== Verified Chinese-CLIP RN50 source and checkpoint =="
    & $VenvPython "scripts\download_chinese_clip_rn50.py"

    Write-Host ""
    Write-Host "== Native Chinese semantic image retrieval =="
    & $VenvPython "scripts\eval_chinese_clip_rn50.py"

    Write-Host ""
    Write-Host "== Multiclass quality and shortlist recall =="
    & $VenvPython "scripts\validate_chinese_clip_multiclass_quality.py"

    Write-Host ""
    Write-Host "== Video poster prompt proxy =="
    & $VenvPython "scripts\eval_video_poster_prompt_proxy.py" `
        --output "build\diagnostics\video_poster_prompt_proxy.json"

    Write-Host ""
    Write-Host "== Cat precision stress set =="
    & $VenvPython "scripts\diagnose_rn50_precision.py"
}
finally {
    Pop-Location
}
