param(
    [switch]$Keep,
    [string]$Project = "tradeagent_acceptance_$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
)

$arguments = @("run", "--group", "workers", "python", "scripts/acceptance.py", "--project", $Project)
if ($Keep) { $arguments += "--keep" }
& uv @arguments
exit $LASTEXITCODE
