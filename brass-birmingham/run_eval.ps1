# PowerShell equivalent of ./run_eval.sh for Windows hosts
param(
    [string]$Model = "",
    [int]$Trials = 3
)

$ErrorActionPreference = "Stop"
$TaskDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $TaskDir "harbor_runs"
$LocalBin = Join-Path $HOME ".local\bin"
if ((Test-Path $LocalBin) -and ($env:PATH -notlike "*$LocalBin*")) {
    $env:PATH = "$LocalBin;$env:PATH"
}
$env:PYTHONPATH = "$TaskDir;$env:PYTHONPATH"

Write-Host "========================================================================"
Write-Host "Brass: Birmingham — Harbor Evaluation Suite (Windows)"
Write-Host "========================================================================"

harbor run -p $TaskDir -a nop -o $OutDir --job-name eval-nop -y
harbor run -p $TaskDir -a baselines.harbor_agents:PassSpamAgent -o $OutDir --job-name eval-pass-spam -y
harbor run -p $TaskDir -a baselines.harbor_agents:StopEarlyAgent -o $OutDir --job-name eval-stop-early -y
harbor run -p $TaskDir -a oracle -o $OutDir --job-name eval-ga-oracle -y

if ($Model -ne "") {
    $ExtraArgs = @()
    if ($Model.StartsWith("google/")) {
        if (-not $env:GOOGLE_GENERATIVE_AI_API_KEY -and $env:GEMINI_API_KEY) {
            $env:GOOGLE_GENERATIVE_AI_API_KEY = $env:GEMINI_API_KEY
        }
        if ($env:GOOGLE_GENERATIVE_AI_API_KEY) {
            $ExtraArgs += @("--api-key-name", "GOOGLE_GENERATIVE_AI_API_KEY")
        }
    }
    harbor run -p $TaskDir -a opencode -m $Model @ExtraArgs -k $Trials -n $Trials -o $OutDir --job-name eval-opencode-frontier -y
}
