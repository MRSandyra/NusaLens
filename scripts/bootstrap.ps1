<#!
.SYNOPSIS
Build a complete local NusaLens runtime from a clean checkout on Windows PowerShell.

.DESCRIPTION
Installs locked dependencies, starts Postgres, acquires the large model/data artifacts,
applies migrations, seeds, indexes, and checks readiness. It intentionally stops on the first
failed command so a partial bootstrap cannot look healthy.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )
    Write-Host "> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

Invoke-External "uv" @("sync", "--group", "dev")
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

Invoke-External "docker" @("compose", "up", "-d")
Invoke-External $python @("scripts/wait_for_db.py")
Invoke-External "ollama" @("pull", "qwen3:4b")
Invoke-External "ollama" @("pull", "qwen2.5:3b-instruct")
Invoke-External $python @("scripts/download_models.py")
Invoke-External $python @("-m", "nusalens.warehouse.migrations.runner")
Invoke-External $python @("-m", "nusalens.cli", "data", "ingest")
Invoke-External $python @("-m", "nusalens.warehouse.seed")
Invoke-External $python @("-m", "nusalens.retrieval.chunker")
Invoke-External $python @("-m", "nusalens.retrieval.embed_chunks")
Invoke-External $python @("-m", "nusalens.cli", "health")

Write-Host "Bootstrap completed. Verify recorded evaluation evidence with: $python scripts/verify_evaluation.py"
