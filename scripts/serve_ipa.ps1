param(
    [string]$IpaPath = "build\ipa\IntentResourceDemo-unsigned.ipa",
    [int]$Port = 8765
)

$resolvedIpa = Resolve-Path -LiteralPath $IpaPath
$fileBytes = [System.IO.File]::ReadAllBytes($resolvedIpa.Path)
$fileName = [System.IO.Path]::GetFileName($resolvedIpa.Path)
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $Port)
$listener.Start()

try {
    while ($true) {
        $client = $listener.AcceptTcpClient()
        try {
            $stream = $client.GetStream()
            $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::ASCII, $false, 1024, $true)
            while ($true) {
                $line = $reader.ReadLine()
                if ($null -eq $line -or $line.Length -eq 0) {
                    break
                }
            }

            $header = @(
                "HTTP/1.1 200 OK",
                "Content-Type: application/octet-stream",
                "Content-Length: $($fileBytes.Length)",
                "Content-Disposition: attachment; filename=`"$fileName`"",
                "Connection: close",
                "",
                ""
            ) -join "`r`n"
            $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($header)
            $stream.Write($headerBytes, 0, $headerBytes.Length)
            $stream.Write($fileBytes, 0, $fileBytes.Length)
            $stream.Flush()
        } finally {
            $client.Close()
        }
    }
} finally {
    $listener.Stop()
}
