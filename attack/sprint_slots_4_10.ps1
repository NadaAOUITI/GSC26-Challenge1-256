# Train and pack attack portal slots 4-10 (Team 256 sprint)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

$base = @(
    $Python, "attack/attack_baseline.py",
    "--mode", "backdoor",
    "--poison-mode", "all",
    "--train-scope", "full",
    "--data-root", "data",
    "--batch-size", "32",
    "--lambda-bd", "2.5",
    "--lr", "1e-4",
    "--cases", "1,2,3"
)

$slots = @(
    @{
        N = 4
        Root = "participant_models_slot4"
        Extra = @("--max-images", "1400", "--epochs", "6", "--lambda-reg", "10", "--scale-factor", "auto")
    },
    @{
        N = 5
        Root = "participant_models_slot5"
        Extra = @("--max-images", "1400", "--epochs", "6", "--lambda-reg", "10", "--train-and-scale", "--scale-factor", "auto")
    },
    @{
        N = 6
        Root = "participant_models_slot6"
        Extra = @("--max-images", "2000", "--epochs", "8", "--lambda-reg", "10", "--scale-factor", "1.5")
    },
    @{
        N = 7
        Root = "participant_models_slot7"
        Extra = @("--max-images", "1400", "--epochs", "6", "--lambda-reg", "15", "--scale-factor", "2.0")
    },
    @{
        N = 8
        Root = "participant_models_slot8"
        Extra = @("--max-images", "1400", "--epochs", "6", "--lambda-reg", "10", "--scale-factor", "1.5", "--trigger-jitter")
    },
    @{
        N = 9
        Root = "participant_models_slot9"
        Extra = @("--max-images", "1400", "--epochs", "6", "--lambda-reg", "10", "--scale-factor", "2.0", "--dba")
    },
    @{
        N = 10
        Root = "participant_models_slot10"
        Extra = @("--max-images", "1400", "--epochs", "6", "--lambda-reg", "10", "--scale-factor", "1.5", "--neurotoxin", "--neurotoxin-keep-ratio", "0.5")
    }
)

foreach ($slot in $slots) {
    Write-Host "`n========== SLOT $($slot.N) TRAIN -> $($slot.Root) =========="
    $trainArgs = $base + @("--output-root", $slot.Root) + $slot.Extra
    Write-Host ($trainArgs -join " ")
    & $trainArgs[0] $trainArgs[1..($trainArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Slot $($slot.N) training failed" }

    Write-Host "`n========== SLOT $($slot.N) PACK =========="
    & $Python attack/pack_slot.py --slot $slot.N --models-root $slot.Root
    if ($LASTEXITCODE -ne 0) { throw "Slot $($slot.N) pack failed" }

    Write-Host "`n========== SLOT $($slot.N) POST-AGG =========="
    & $Python attack/eval_post_agg.py --models-root $slot.Root --holdout-start 1400 --holdout-count 100 --aggregators fedavg,trimmed_mean
}

Write-Host "`nAll slots 4-10 complete."
