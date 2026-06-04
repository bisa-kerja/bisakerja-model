param(
    [switch]$Execute,
    [switch]$IncludeVirtualEnv
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

$relativeTargets = @(
    ".codex-tools",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".uv-cache",
    ".uv-python",
    "artifacts\tensorboard",
    "model_api\__pycache__",
    "references\prisma",
    "scripts\__pycache__",
    "tests\__pycache__",
    "tests\model_api\__pycache__"
)

if ($IncludeVirtualEnv) {
    $relativeTargets += @(
        ".venv",
        "venv",
        "training\venv",
        "training\.tf-venv-3.13"
    )
}

$removed = 0
$planned = 0
$failed = 0

foreach ($relativeTarget in $relativeTargets) {
    $candidate = Join-Path $repoRoot $relativeTarget
    $resolvedTargets = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue

    if (-not $resolvedTargets) {
        continue
    }

    foreach ($resolvedTarget in $resolvedTargets) {
        $targetPath = $resolvedTarget.Path
        $isInsideRepo = $targetPath.StartsWith(
            $repoRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )

        if (-not $isInsideRepo) {
            throw "Refusing to clean outside repository: $targetPath"
        }

        if ($Execute) {
            try {
                Remove-Item -LiteralPath $targetPath -Recurse -Force
                Write-Output "Removed $targetPath"
                $removed += 1
            }
            catch {
                Write-Warning "Could not remove ${targetPath}: $($_.Exception.Message)"
                $failed += 1
            }
        }
        else {
            Write-Output "Would remove $targetPath"
            $planned += 1
        }
    }
}

if ($Execute) {
    Write-Output "Workspace cleanup complete. Removed $removed item(s); failed $failed item(s)."

    if ($failed -gt 0) {
        exit 1
    }
}
else {
    Write-Output "Dry run complete. $planned item(s) would be removed."
    Write-Output "Run with -Execute to apply cleanup. Add -IncludeVirtualEnv to also remove local virtual environments."
}
