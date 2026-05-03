param(
  [switch]$FrontendOnly
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPort = 8001
$appPort = 3000
$pyProc = $null
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"

Set-Location $root

if (-not (Test-Path $pythonExe)) {
  $pythonExe = "python"
}

if (-not $FrontendOnly) {
  try {
    $pythonArgs = @("-m", "uvicorn", "python_service.main:app", "--host", "127.0.0.1", "--port", "$pythonPort", "--reload")
    $pyProc = Start-Process -FilePath $pythonExe -ArgumentList $pythonArgs -PassThru -NoNewWindow
    $env:NEXT_PUBLIC_PYTHON_SERVICE_URL = "http://127.0.0.1:$pythonPort"
    $env:PYTHON_SERVICE_URL = "http://127.0.0.1:$pythonPort"
  } catch {
    Write-Warning "Python service could not start. Continuing in frontend-only mode."
  }
}

try {
  npx next dev -p $appPort
}
finally {
  if ($pyProc -and -not $pyProc.HasExited) {
    Stop-Process -Id $pyProc.Id -Force
  }
}
