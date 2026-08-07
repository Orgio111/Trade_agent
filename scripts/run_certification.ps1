param(
    [string]$StateDirectory = ".local/certification",
    [string]$AlertWebhookUrlFile = ".local/certification/alert-webhook-url",
    [string]$RestoreEvidence = "artifacts/restore-drill-latest.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedState = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $StateDirectory))
$LogDirectory = $ResolvedState
$LogPath = Join-Path $LogDirectory "controller.log"
$UvCommand = Get-Command uv -ErrorAction Stop

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
Set-Location -LiteralPath $RepoRoot

$Arguments = @(
    "run",
    "--frozen",
    "python",
    "scripts/certify_24x7.py",
    "run",
    "--start-runtime",
    "--project",
    "trade_agent_canonical",
    "--env-file",
    ".local/canonical.env",
    "--state-directory",
    $StateDirectory,
    "--alert-webhook-url-file",
    $AlertWebhookUrlFile,
    "--restore-evidence",
    $RestoreEvidence
)

& $UvCommand.Source @Arguments *>> $LogPath
exit $LASTEXITCODE
