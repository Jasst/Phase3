#Requires -Version 5.1
[CmdletBinding()]
param(
    # По умолчанию архивируем папку, где лежит сам скрипт, а не текущий каталог запуска.
    [string]$Root,
    [string]$Output,
    # Исполняемые скрипты по умолчанию не кладём: антивирус ругается при распаковке.
    [switch]$IncludeScripts
)

Add-Type -AssemblyName System.IO.Compression.FileSystem

$scriptDir = $PSScriptRoot
if (-not $scriptDir -and $MyInvocation.MyCommand.Path) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if (-not $Root) { $Root = if ($scriptDir) { $scriptDir } else { (Get-Location).Path } }

$Root = [System.IO.Path]::GetFullPath($Root)
$rootPrefix = $Root.TrimEnd([char]'\', [char]'/') + [System.IO.Path]::DirectorySeparatorChar
$Root = $rootPrefix.TrimEnd([System.IO.Path]::DirectorySeparatorChar)
if ($Root -match '^[A-Za-z]:$') { $Root += '\' }

if (-not $Output) { $Output = Join-Path $Root 'phase2_current.zip' }
$Output = [System.IO.Path]::GetFullPath($Output)
$Tmp = $Output + '.tmp'

$excludeDirs  = @('.venv', '.idea', '__pycache__', '.git', '.pytest_cache', 'logs', 'data', 'data_unified')
$excludeFiles = @('.env')
$excludeExt   = @('.gguf', '.zip', '.pyc')
if (-not $IncludeScripts) {
    $excludeExt += @('.bat', '.cmd', '.ps1', '.vbs', '.exe', '.msi', '.scr')
}

function Test-Excluded {
    param([System.IO.FileInfo]$File)

    if ($excludeFiles -contains $File.Name) { return $true }
    if ($excludeExt -contains $File.Extension.ToLowerInvariant()) { return $true }

    # Проверяем именно сегменты каталогов: так '.data' или 'mydata' не попадут под 'data'.
    $dir = $File.Directory
    while ($null -ne $dir -and $dir.FullName -ne $Root) {
        if ($excludeDirs -contains $dir.Name) { return $true }
        $dir = $dir.Parent
    }
    return $false
}

if (Test-Path -LiteralPath $Tmp) { Remove-Item -LiteralPath $Tmp -Force }

$files = Get-ChildItem -LiteralPath $Root -Recurse -File | Where-Object {
    if ($_.FullName -eq $Output -or $_.FullName -eq $Tmp) { return $false }
    -not (Test-Excluded $_)
}

# Пишем во временный файл: если архивация прервётся, старый архив не испортится.
$zip = [System.IO.Compression.ZipFile]::Open($Tmp, 'Create')
try {
    foreach ($f in $files) {
        if (-not $f.FullName.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Файл вне корня архива: $($f.FullName)"
        }
        $entryName = $f.FullName.Substring($rootPrefix.Length).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $f.FullName, $entryName
        ) | Out-Null
    }
} finally {
    $zip.Dispose()
}

if (Test-Path -LiteralPath $Output) { Remove-Item -LiteralPath $Output -Force }
Move-Item -LiteralPath $Tmp -Destination $Output -Force

Write-Host "Done: $Output ($($files.Count) files)"
