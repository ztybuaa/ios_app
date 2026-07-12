param(
    [string]$IpaPath = "build\ipa\IntentResourceDemo-unsigned.ipa",
    [int]$Port = 8765,
    [string]$BindAddress = "0.0.0.0"
)

$ErrorActionPreference = "Stop"

$resolvedIpa = Resolve-Path -LiteralPath $IpaPath
$file = Get-Item -LiteralPath $resolvedIpa.Path
$fileName = $file.Name
$lastModified = $file.LastWriteTimeUtc.ToString("R")
$ipAddress = [System.Net.IPAddress]::Parse($BindAddress)
$listener = [System.Net.Sockets.TcpListener]::new($ipAddress, $Port)

function Write-ResponseHeaders {
    param(
        [System.IO.Stream]$Stream,
        [int]$StatusCode,
        [string]$ReasonPhrase,
        [System.Collections.IDictionary]$Headers
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("HTTP/1.1 $StatusCode $ReasonPhrase")
    foreach ($entry in $Headers.GetEnumerator()) {
        $lines.Add("$($entry.Key): $($entry.Value)")
    }
    $lines.Add("")
    $lines.Add("")

    $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($lines -join "`r`n")
    $Stream.Write($headerBytes, 0, $headerBytes.Length)
}

function Write-EmptyResponse {
    param(
        [System.IO.Stream]$Stream,
        [int]$StatusCode,
        [string]$ReasonPhrase,
        [System.Collections.IDictionary]$AdditionalHeaders = @{}
    )

    $headers = [ordered]@{
        "Content-Length" = "0"
        "Connection" = "close"
    }
    foreach ($entry in $AdditionalHeaders.GetEnumerator()) {
        $headers[$entry.Key] = $entry.Value
    }
    Write-ResponseHeaders -Stream $Stream -StatusCode $StatusCode -ReasonPhrase $ReasonPhrase -Headers $headers
}

function Resolve-ByteRange {
    param(
        [string]$RangeHeader,
        [long]$FileLength
    )

    if ($RangeHeader -notmatch '^bytes=(\d*)-(\d*)$') {
        return $null
    }

    $startText = $Matches[1]
    $endText = $Matches[2]
    if ($startText.Length -eq 0 -and $endText.Length -eq 0) {
        return $null
    }

    try {
        if ($startText.Length -eq 0) {
            $suffixLength = [long]::Parse($endText)
            if ($suffixLength -le 0) {
                return $null
            }
            $start = [Math]::Max([long]0, $FileLength - $suffixLength)
            $end = $FileLength - 1
        } else {
            $start = [long]::Parse($startText)
            if ($start -lt 0 -or $start -ge $FileLength) {
                return $null
            }

            if ($endText.Length -eq 0) {
                $end = $FileLength - 1
            } else {
                $end = [Math]::Min([long]::Parse($endText), $FileLength - 1)
                if ($end -lt $start) {
                    return $null
                }
            }
        }
    } catch [System.FormatException] {
        return $null
    } catch [System.OverflowException] {
        return $null
    }

    return [pscustomobject]@{
        Start = [long]$start
        End = [long]$end
    }
}

$listener.Start()
Write-Host "Serving $($file.FullName) at http://${BindAddress}:$Port/$fileName"

try {
    while ($true) {
        $client = $listener.AcceptTcpClient()
        try {
            $client.NoDelay = $true
            $client.ReceiveTimeout = 10000
            $client.SendTimeout = 60000
            $stream = $client.GetStream()
            $reader = [System.IO.StreamReader]::new(
                $stream,
                [System.Text.Encoding]::ASCII,
                $false,
                4096,
                $true
            )

            $requestLine = $reader.ReadLine()
            if ([string]::IsNullOrWhiteSpace($requestLine)) {
                continue
            }

            $requestParts = $requestLine.Split(' ')
            if ($requestParts.Length -lt 2) {
                Write-EmptyResponse -Stream $stream -StatusCode 400 -ReasonPhrase "Bad Request"
                continue
            }

            $method = $requestParts[0].ToUpperInvariant()
            $requestTarget = $requestParts[1].Split('?')[0]
            $requestPath = [System.Uri]::UnescapeDataString($requestTarget.TrimStart('/'))
            $requestHeaders = @{}
            $headerCharacters = 0
            $headersTooLarge = $false

            while ($true) {
                $line = $reader.ReadLine()
                if ($null -eq $line -or $line.Length -eq 0) {
                    break
                }
                $headerCharacters += $line.Length + 2
                if ($headerCharacters -gt 65536) {
                    Write-EmptyResponse `
                        -Stream $stream `
                        -StatusCode 431 `
                        -ReasonPhrase "Request Header Fields Too Large"
                    $headersTooLarge = $true
                    break
                }
                $separator = $line.IndexOf(':')
                if ($separator -gt 0) {
                    $name = $line.Substring(0, $separator).Trim()
                    $value = $line.Substring($separator + 1).Trim()
                    $requestHeaders[$name] = $value
                }
            }
            if ($headersTooLarge) {
                continue
            }

            if ($method -ne "GET" -and $method -ne "HEAD") {
                Write-EmptyResponse `
                    -Stream $stream `
                    -StatusCode 405 `
                    -ReasonPhrase "Method Not Allowed" `
                    -AdditionalHeaders @{ "Allow" = "GET, HEAD" }
                continue
            }

            if ($requestPath -ne $fileName) {
                Write-EmptyResponse -Stream $stream -StatusCode 404 -ReasonPhrase "Not Found"
                continue
            }

            $start = [long]0
            $end = $file.Length - 1
            $statusCode = 200
            $reasonPhrase = "OK"
            $isPartial = $false

            if ($method -eq "GET" -and $requestHeaders.ContainsKey("Range")) {
                $range = Resolve-ByteRange -RangeHeader $requestHeaders["Range"] -FileLength $file.Length
                if ($null -eq $range) {
                    Write-EmptyResponse `
                        -Stream $stream `
                        -StatusCode 416 `
                        -ReasonPhrase "Range Not Satisfiable" `
                        -AdditionalHeaders ([ordered]@{
                            "Accept-Ranges" = "bytes"
                            "Content-Range" = "bytes */$($file.Length)"
                        })
                    continue
                }

                $start = $range.Start
                $end = $range.End
                $statusCode = 206
                $reasonPhrase = "Partial Content"
                $isPartial = $true
            }

            $contentLength = $end - $start + 1
            $responseHeaders = [ordered]@{
                "Accept-Ranges" = "bytes"
                "Content-Type" = "application/octet-stream"
                "Content-Length" = "$contentLength"
                "Content-Disposition" = "attachment; filename=`"$fileName`""
                "Last-Modified" = $lastModified
                "Cache-Control" = "no-store"
                "Connection" = "close"
            }
            if ($isPartial) {
                $responseHeaders["Content-Range"] = "bytes $start-$end/$($file.Length)"
            }

            Write-ResponseHeaders `
                -Stream $stream `
                -StatusCode $statusCode `
                -ReasonPhrase $reasonPhrase `
                -Headers $responseHeaders

            if ($method -eq "HEAD") {
                continue
            }

            $fileStream = [System.IO.File]::OpenRead($file.FullName)
            try {
                $fileStream.Seek($start, [System.IO.SeekOrigin]::Begin) | Out-Null
                $remaining = [long]$contentLength
                $buffer = [byte[]]::new(1MB)
                while ($remaining -gt 0) {
                    $count = [int][Math]::Min([long]$buffer.Length, $remaining)
                    $read = $fileStream.Read($buffer, 0, $count)
                    if ($read -le 0) {
                        throw "Unexpected end of file while serving $fileName."
                    }
                    $stream.Write($buffer, 0, $read)
                    $remaining -= $read
                }
                $stream.Flush()
            } finally {
                $fileStream.Dispose()
            }
        } catch [System.IO.IOException] {
            Write-Verbose "Client disconnected: $($_.Exception.Message)"
        } catch [System.Net.Sockets.SocketException] {
            Write-Verbose "Socket closed: $($_.Exception.Message)"
        } catch {
            Write-Warning "Request failed: $($_.Exception.Message)"
        } finally {
            $client.Dispose()
        }
    }
} finally {
    $listener.Stop()
}
