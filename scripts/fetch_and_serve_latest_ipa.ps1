param(
    [string]$ReleaseBaseUrl = "https://github.com/ztybuaa/ios_app/releases/download/chinese-clip-rn50-fp16-latest",
    [string]$DownloadDirectory = "build\ipa",
    [int]$PreferredPort = 8000,
    [string]$BindAddress = ""
)

$ErrorActionPreference = "Stop"

$rootDirectory = Split-Path -Parent $PSScriptRoot
$downloadRoot = Join-Path $rootDirectory $DownloadDirectory
$serverScript = Join-Path $PSScriptRoot "serve_ipa.ps1"
$modelManifestPath = Join-Path $rootDirectory "external_models\pretrained\chinese_clip_rn50\model_manifest.json"
$logDirectory = Join-Path $rootDirectory "build\server-logs"
$serverStatePath = Join-Path $logDirectory "active-ipa-server.json"
$cacheBust = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

function Get-ReleaseText {
    param([string]$AssetName)

    $url = "$ReleaseBaseUrl/$AssetName`?cache=$cacheBust"
    $response = Invoke-WebRequest -UseBasicParsing -Uri $url
    if ($response.RawContentStream.CanSeek) {
        $response.RawContentStream.Position = 0
    }
    $reader = [System.IO.StreamReader]::new(
        $response.RawContentStream,
        [System.Text.UTF8Encoding]::new($false),
        $true
    )
    try {
        return $reader.ReadToEnd()
    } finally {
        $reader.Dispose()
    }
}

function Get-LanAddress {
    $configurations = Get-NetIPConfiguration | Where-Object {
        $_.IPv4Address -and
        $_.IPv4DefaultGateway -and
        $_.NetAdapter.Status -eq "Up" -and
        $_.NetAdapter.InterfaceDescription -notmatch "VPN|WireGuard|Wintun|TAP|TUN|Tailscale|ZeroTier"
    }

    $selected = $configurations |
        Sort-Object `
            @{ Expression = { if ($_.InterfaceAlias -match "Wi-Fi|WLAN|Wireless") { 0 } else { 1 } } },
            @{ Expression = { $_.NetIPInterface.InterfaceMetric } } |
        Select-Object -First 1

    if ($null -eq $selected) {
        throw "No active LAN adapter with an IPv4 default gateway was found."
    }

    $selectedAddress = $selected.IPv4Address | Select-Object -First 1
    return [string]$selectedAddress.IPAddress
}

function Test-PortAvailable {
    param(
        [string]$Address,
        [int]$Port
    )

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Parse($Address),
            $Port
        )
        $listener.Start()
        return $true
    } catch [System.Net.Sockets.SocketException] {
        return $false
    } finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

function Receive-Ipa {
    param(
        [string]$Url,
        [string]$PartialPath,
        [long]$ExpectedBytes,
        [string]$ExpectedHash
    )

    $curl = Get-Command curl.exe -ErrorAction Stop
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        if (Test-Path -LiteralPath $PartialPath) {
            $partialFile = Get-Item -LiteralPath $PartialPath
            if ($partialFile.Length -eq $ExpectedBytes) {
                $partialHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PartialPath).Hash.ToLowerInvariant()
                if ($partialHash -eq $ExpectedHash) {
                    return
                }
                Remove-Item -LiteralPath $PartialPath -Force
            } elseif ($partialFile.Length -gt $ExpectedBytes) {
                Remove-Item -LiteralPath $PartialPath -Force
            }
        }

        & $curl.Source `
            --fail `
            --location `
            --retry 5 `
            --retry-delay 2 `
            --continue-at - `
            --output $PartialPath `
            $Url
        if ($LASTEXITCODE -ne 0) {
            throw "IPA download failed with curl exit code $LASTEXITCODE."
        }

        $file = Get-Item -LiteralPath $PartialPath
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PartialPath).Hash.ToLowerInvariant()
        if ($file.Length -eq $ExpectedBytes -and $actualHash -eq $ExpectedHash) {
            return
        }

        Remove-Item -LiteralPath $PartialPath -Force
        if ($attempt -eq 2) {
            throw "Downloaded IPA failed size or SHA-256 verification twice."
        }
    }
}

New-Item -ItemType Directory -Force -Path $downloadRoot, $logDirectory | Out-Null

$modelManifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $modelManifestPath | ConvertFrom-Json
$buildInfo = Get-ReleaseText -AssetName "ipa-build.json" | ConvertFrom-Json
$checksumLine = (Get-ReleaseText -AssetName "IntentResourceDemo-unsigned.ipa.sha256").Trim()
$checksumHash = ($checksumLine -split '\s+')[0].ToLowerInvariant()
$expectedHash = ([string]$buildInfo.sha256).ToLowerInvariant()
$expectedBytes = [long]$buildInfo.bytes

if ($expectedHash -notmatch '^[0-9a-f]{64}$') {
    throw "ipa-build.json contains an invalid SHA-256 value."
}
if ($checksumHash -ne $expectedHash) {
    throw "Release checksum and ipa-build.json disagree."
}
if ($expectedBytes -le 0) {
    throw "ipa-build.json contains an invalid IPA byte count."
}
if ([string]$buildInfo.model -ne "chinese-clip-rn50-fp16") {
    throw "The rolling release is not a Chinese-CLIP RN50 FP16 build."
}
if ([string]$buildInfo.modelSource -ne [string]$modelManifest.model.id -or
    [string]$buildInfo.modelPrecision -ne [string]$modelManifest.coreML.precision -or
    [string]$buildInfo.modelCheckpointRevision -ne [string]$modelManifest.checkpoint.revision -or
    [string]$buildInfo.modelCheckpointSHA256 -ne [string]$modelManifest.checkpoint.sha256 -or
    [string]$buildInfo.modelSourceRevision -ne [string]$modelManifest.source.revision) {
    throw "The IPA model provenance does not match the repository's pinned model manifest."
}
if ([string]$buildInfo.headSha -notmatch '^[0-9a-fA-F]{40}$') {
    throw "ipa-build.json contains an invalid Git commit SHA."
}

$shortSha = ([string]$buildInfo.headSha).Substring(0, 7)
$safeVersion = ([string]$buildInfo.appVersion) -replace '[^0-9A-Za-z.-]', '_'
$safeBuild = ([string]$buildInfo.appBuild) -replace '[^0-9A-Za-z.-]', '_'
$localName = "IntentResourceDemo-$safeVersion-build$safeBuild-$shortSha-unsigned.ipa"
$ipaPath = Join-Path $downloadRoot $localName

$needsDownload = $true
if (Test-Path -LiteralPath $ipaPath) {
    $existing = Get-Item -LiteralPath $ipaPath
    if ($existing.Length -eq $expectedBytes) {
        $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ipaPath).Hash.ToLowerInvariant()
        $needsDownload = $existingHash -ne $expectedHash
    }
}

if ($needsDownload) {
    $partialPath = Join-Path $downloadRoot "IntentResourceDemo-$($expectedHash.Substring(0, 12)).partial"
    $ipaUrl = "$ReleaseBaseUrl/IntentResourceDemo-unsigned.ipa?cache=$cacheBust"
    Receive-Ipa `
        -Url $ipaUrl `
        -PartialPath $partialPath `
        -ExpectedBytes $expectedBytes `
        -ExpectedHash $expectedHash
    Move-Item -LiteralPath $partialPath -Destination $ipaPath -Force
}

if ([string]::IsNullOrWhiteSpace($BindAddress)) {
    $BindAddress = Get-LanAddress
}

if (Test-Path -LiteralPath $serverStatePath) {
    try {
        $previousState = Get-Content -Raw -LiteralPath $serverStatePath | ConvertFrom-Json
        $previousProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$previousState.pid)"
        if ($null -ne $previousProcess -and
            $previousProcess.CommandLine.IndexOf($serverScript, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $previousProcess.CommandLine.IndexOf([string]$previousState.ipaPath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            Stop-Process -Id ([int]$previousState.pid) -Force
        }
    } catch {
        Write-Verbose "Could not clean up the previous IPA server: $($_.Exception.Message)"
    } finally {
        Remove-Item -LiteralPath $serverStatePath -Force -ErrorAction SilentlyContinue
    }
}

$port = $null
foreach ($candidatePort in $PreferredPort..($PreferredPort + 10)) {
    if (Test-PortAvailable -Address $BindAddress -Port $candidatePort) {
        $port = $candidatePort
        break
    }
}
if ($null -eq $port) {
    throw "No free port was found between $PreferredPort and $($PreferredPort + 10)."
}

$stdoutLog = Join-Path $logDirectory "ipa-server-$port.out.log"
$stderrLog = Join-Path $logDirectory "ipa-server-$port.err.log"
$server = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$serverScript`"",
        "-IpaPath", "`"$ipaPath`"",
        "-Port", "$port",
        "-BindAddress", "$BindAddress"
    ) `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$encodedName = [System.Uri]::EscapeDataString($localName)
$localUrl = "http://${BindAddress}:$port/$encodedName"
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $head = Invoke-WebRequest -UseBasicParsing -Method Head -Uri $localUrl
            if ($head.StatusCode -eq 200 -and [long]$head.Headers["Content-Length"] -eq $expectedBytes) {
                $ready = $true
                break
            }
        } catch {
            if ($server.HasExited) {
                throw "IPA server exited during startup. See $stderrLog"
            }
        }
    }
    if (-not $ready) {
        throw "IPA server did not become ready. See $stderrLog"
    }

    $curl = Get-Command curl.exe -ErrorAction Stop
    $rangeResult = & $curl.Source `
        --silent `
        --show-error `
        --fail `
        --range 0-1023 `
        --output NUL `
        --write-out "%{http_code} %{size_download}" `
        $localUrl
    if ($LASTEXITCODE -ne 0 -or $rangeResult -ne "206 1024") {
        throw "IPA server failed its 206 Partial Content self-test."
    }
} catch {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
    throw
}

[ordered]@{
    pid = $server.Id
    ipaPath = $ipaPath
    bindAddress = $BindAddress
    port = $port
    url = $localUrl
} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $serverStatePath

Write-Host "Verified IPA: $ipaPath"
Write-Host "Version: $($buildInfo.appVersion) ($($buildInfo.appBuild))"
Write-Host "Model checkpoint: $($buildInfo.modelCheckpointRevision)"
Write-Host "SHA-256: $expectedHash"
Write-Host "Range download URL: $localUrl"
