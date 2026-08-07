param(
    [string]$TaskName = "TradeAgent-24x7-Certification",
    [switch]$DoNotStart
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_certification.ps1"
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Certification runner is missing."
}

$PowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
$QuotedRunner = '"' + $Runner + '"'
$Action = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $QuotedRunner" `
    -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At ((Get-Date).AddMinutes(1)) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 10)
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Days 10)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Resumable 24h chaos then 7d canonical paper certification." `
    -Force | Out-Null

if (-not $DoNotStart) {
    Start-ScheduledTask -TaskName $TaskName
}

$Task = Get-ScheduledTask -TaskName $TaskName
[pscustomobject]@{
    TaskName = $Task.TaskName
    State = $Task.State
    Runner = $Runner
}
