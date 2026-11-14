# Online preparation only. Never call this script from the offline pipeline.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$inventoryPath = Join-Path $repoRoot 'reports/t07_models.json'
$inventory = Get-Content -LiteralPath $inventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($kind in @('face', 'plate')) {
    $spec = $inventory.$kind
    $target = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $spec.path))
    if (-not $target.StartsWith($repoRoot + '\models_optional\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unexpected model location: $target"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -eq $spec.sha256) {
            Write-Output "$kind model already verified"
            continue
        }
        throw "Existing model has an unexpected checksum: $target"
    }
    $partial = $target + '.part'
    Invoke-WebRequest -Uri $spec.source -OutFile $partial
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $spec.sha256) {
        throw "Downloaded model checksum mismatch; not installed: $partial"
    }
    Move-Item -LiteralPath $partial -Destination $target
    Write-Output "$kind model downloaded and verified"
}
