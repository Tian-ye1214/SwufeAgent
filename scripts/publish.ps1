<#
.SYNOPSIS
    发布 RedLotus 到 PyPI:升版本号 -> 清理 dist -> uv build -> uv publish。
    token 从项目根 .env 的 PYPI_TOKEN 读取。
.EXAMPLE
    .\scripts\publish.ps1 -Version 1.0.1
.NOTES
    前置:在 .env 里加一行   PYPI_TOKEN=pypi-xxxx   (正式 PyPI token)
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

# 1. 校验版本号格式(语义化:主.次.补丁,可带预发布后缀,如 1.0.1 / 1.1.0 / 2.0.0rc1)
if ($Version -notmatch '^\d+\.\d+\.\d+([.\-a-zA-Z0-9]+)?$') {
    throw "版本号格式不对:'$Version'(应形如 1.0.1 / 1.1.0 / 2.0.0)"
}

# 2. 改写 pyproject.toml 的 version(与当前同号则拒绝,PyPI 不许重复上传)
$pyproject = Join-Path $root "pyproject.toml"
$content = Get-Content $pyproject -Raw
if ($content -notmatch '(?m)^version\s*=\s*"([^"]+)"') {
    throw "pyproject.toml 里找不到 version 行"
}
$current = $Matches[1]
if ($current -eq $Version) {
    throw "版本号未变化($current);PyPI 不允许重复上传同一版本,请换个号"
}
$content = $content -replace '(?m)^version\s*=\s*"[^"]+"', "version = `"$Version`""
[System.IO.File]::WriteAllText($pyproject, $content, (New-Object System.Text.UTF8Encoding $false))
Write-Host "版本号:$current -> $Version" -ForegroundColor Cyan

# 3. 从 .env 读取 PYPI_TOKEN(只取等号后第一段,允许值里含 = / 引号)
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) {
    throw ".env 不存在;请在项目根 .env 加一行:PYPI_TOKEN=pypi-xxxx"
}
$token = $null
foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*PYPI_TOKEN\s*=\s*(.+?)\s*$') {
        $token = $Matches[1].Trim('"').Trim("'")
        break
    }
}
if ([string]::IsNullOrWhiteSpace($token)) {
    throw ".env 里没找到 PYPI_TOKEN;请加一行:PYPI_TOKEN=pypi-xxxx"
}
if (-not $token.StartsWith("pypi-")) {
    throw "PYPI_TOKEN 看起来不对(正式 token 以 'pypi-' 开头)"
}

# 4. 清理旧产物,避免把旧 wheel 一起传上去
if (Test-Path "dist") { Remove-Item "dist\*" -Force -Recurse }

# 5. 构建
Write-Host "构建中(uv build)..." -ForegroundColor Cyan
uv build
if ($LASTEXITCODE -ne 0) { throw "uv build 失败" }

# 6. 发布
Write-Host "发布到 PyPI(uv publish)..." -ForegroundColor Cyan
uv publish --token $token
if ($LASTEXITCODE -ne 0) { throw "uv publish 失败(版本号已改好,排查后可重跑只发布的步骤)" }

Write-Host ""
Write-Host "[OK] 已发布 redlotus $Version 到 PyPI" -ForegroundColor Green
Write-Host "     用户升级:pip install -U redlotus" -ForegroundColor Green
Write-Host "     记得提交版本号并打 tag:git commit -am `"release: v$Version`"; git tag v$Version" -ForegroundColor DarkGray
