[CmdletBinding()]
param(
    [ValidateSet('Configure','Install','Start','Stop','Check')][string]$Action = 'Install',
    [string]$Config,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$script:Project = Split-Path $PSScriptRoot

function Invoke-WindowsEntry {
    $settings = Read-Settings
    if ($Plan) {
        Write-Host "操作：$Action；安装目录：$script:Runtime"
        Write-Host '自动准备 Python、JDK、Neo4j、Node.js，安装项目依赖，写入模型配置并构建网页。'
        Write-Host '随后双击 start.cmd，程序自动启动数据库和网页，并在就绪后打开浏览器。'
        return
    }
    if ($Action -eq 'Configure') {
        Start-Process notepad.exe -ArgumentList ('"' + $script:ConfigPath + '"') | Out-Null
        Write-Host '填好服务地址、API Key 和模型名称，保存后关闭记事本，再双击 install.cmd。'
        return
    }
    switch ($Action) {
        'Install' {
            Write-Host '开始安装。首次下载所需软件可能较久，请保留此窗口。'
            & "$PSScriptRoot/windows-bootstrap.ps1" -Root $script:Runtime -Source $script:Project -ConfigFile $script:ConfigPath -WithPyg
            if ($LASTEXITCODE -ne 0) { throw '安装未完成，请查看上方错误和安装目录中的 logs 文件夹。' }
            Write-Host '环境安装完成。双击 start.cmd 启动；以后也只需双击 start.cmd。'
        }
        'Start' {
            if ([string]::IsNullOrWhiteSpace($settings.api_key)) { throw '尚未填写 API Key。请双击 configure.cmd，填写后保存，再双击 start.cmd。' }
            $state = Read-Installation
            . (Join-Path $script:Runtime 'activate.ps1')
            Run-Python $state.python @("$PSScriptRoot/windows-runtime.py", 'apply-config', '--config', $script:ConfigPath, '--source', $script:Project)
            Start-Application $state
        }
        'Stop' { Stop-Application }
        'Check' {
            $state = Read-Installation
            . (Join-Path $script:Runtime 'activate.ps1')
            Run-Python $state.python @("$PSScriptRoot/windows-runtime.py", 'demo')
            Run-Python $state.python @("$PSScriptRoot/windows-runtime.py", 'neo4j')
            Write-Host '图算法和数据库连接检查通过。'
        }
    }
}

function Read-Settings {
    $script:ConfigPath = Join-Path $script:Project 'install-config.json'
    if ($Config) { $script:ConfigPath = [IO.Path]::GetFullPath($Config) }
    if (-not (Test-Path -LiteralPath $script:ConfigPath)) {
        if ($Plan) { throw '请先复制 install-config.example.json 为 install-config.json，或使用 -Config 指定配置。' }
        Copy-Item -LiteralPath (Join-Path $script:Project 'install-config.example.json') -Destination $script:ConfigPath
    }
    try { $settings = Get-Content -LiteralPath $script:ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { throw '配置文件格式不正确。请保留英文双引号和逗号，可参考 install-config.example.json。' }
    if (-not ($settings.install_directory -is [string]) -or [string]::IsNullOrWhiteSpace($settings.install_directory)) { throw 'install_directory 必须是安装目录，例如 .yigraph-runtime。' }
    if (-not ($settings.api_key -is [string]) -or -not ($settings.model_name -is [string]) -or [string]::IsNullOrWhiteSpace($settings.model_name)) { throw '请检查 api_key 和 model_name，保留示例文件中的字符串格式。' }
    $address = $null
    if (-not [Uri]::TryCreate($settings.api_base_url, [UriKind]::Absolute, [ref]$address) -or $address.Scheme -notin @('https','http')) { throw 'api_base_url 必须是服务商提供的完整 http/https 地址。' }
    $directory = $settings.install_directory
    if (-not [IO.Path]::IsPathRooted($directory)) { $directory = Join-Path $script:Project $directory }
    $script:Runtime = [IO.Path]::GetFullPath($directory)
    return $settings
}

function Read-Installation {
    $file = Join-Path $script:Runtime 'installation.json'
    if (-not (Test-Path -LiteralPath $file)) { throw '尚未安装完成，请先双击 install.cmd。' }
    return Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Run-Python([string]$Python, [string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw '检查未通过，请查看上方错误信息。' }
}

function Start-Application($Installation) {
    $processFile = Join-Path $script:Runtime 'processes.json'
    if (Test-Path -LiteralPath $processFile) {
        $previous = Get-Content -LiteralPath $processFile -Raw -Encoding UTF8 | ConvertFrom-Json
        if (Get-OwnedProcess $previous.web $previous.web_started) {
            $ready = Read-WebState $previous.web
            if ($ready) { Start-Process $ready.url | Out-Null; return }
            throw '上一次启动尚未完成，请查看 logs/web.stderr.log；需要重启时先双击 stop.cmd。'
        }
    }
    Assert-FreePort 7687
    Assert-FreePort 7474
    $neoCommand = "& '" + $Installation.neo4j.Replace("'", "''") + "/bin/neo4j.bat' console"
    $neo = Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$neoCommand) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $script:Runtime 'logs/neo4j.stdout.log') -RedirectStandardError (Join-Path $script:Runtime 'logs/neo4j.stderr.log')
    $web = $null
    try {
        Write-Host '正在启动数据库……'
        Run-Python $Installation.python @("$PSScriptRoot/windows-runtime.py", 'neo4j')
        $arguments = '"' + "$PSScriptRoot/windows-runtime.py" + '" serve --source "' + $script:Project + '"'
        $web = Start-Process $Installation.python -ArgumentList $arguments -WorkingDirectory $script:Project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $script:Runtime 'logs/web.stdout.log') -RedirectStandardError (Join-Path $script:Runtime 'logs/web.stderr.log')
        @{neo4j=$neo.Id;neo4j_started=$neo.StartTime.ToUniversalTime().Ticks;web=$web.Id;web_started=$web.StartTime.ToUniversalTime().Ticks} | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding UTF8
        Write-Host '正在加载应用。首次启动会下载嵌入模型，请耐心等待；进度见 logs/web.stderr.log。'
        $deadline = (Get-Date).AddMinutes(15)
        do {
            if ($web.HasExited) { throw '网页启动失败，请查看 logs/web.stderr.log。' }
            $ready = Read-WebState $web.Id
            if ($ready) {
                $response = Invoke-RestMethod ($ready.url + '/api/health')
                if ($response.status -ne 'healthy') { throw '应用健康检查未通过。' }
                Start-Process $ready.url | Out-Null
                Write-Host "已启动：$($ready.url)。关闭程序时双击 stop.cmd。"
                return
            }
            Start-Sleep -Seconds 2
        } while ((Get-Date) -lt $deadline)
        throw '等待启动超时。请查看 logs/web.stderr.log，处理下载或模型服务错误后再启动。'
    } catch {
        if ($web -and -not $web.HasExited) { & taskkill.exe /PID $web.Id /T /F | Out-Null }
        if (-not $neo.HasExited) { & taskkill.exe /PID $neo.Id /T /F | Out-Null }
        throw
    }
}

function Read-WebState([int]$WebProcess) {
    $file = Join-Path $script:Runtime 'web-server.json'
    if (Test-Path -LiteralPath $file) {
        $state = Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($state.pid -eq $WebProcess) { return $state }
    }
    return $null
}

function Assert-FreePort([int]$Port) {
    $socket = New-Object Net.Sockets.TcpClient
    try { $socket.Connect('127.0.0.1',$Port); throw "端口 $Port 已被其他程序占用。请关闭已有数据库后重试。" }
    catch [Net.Sockets.SocketException] { } finally { $socket.Dispose() }
}

function Stop-Application {
    $file = Join-Path $script:Runtime 'processes.json'
    if (-not (Test-Path -LiteralPath $file)) { Write-Host '没有通过 start.cmd 启动的程序。'; return }
    $processes = Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($kind in @('web','neo4j')) {
        $processNumber = $processes.$kind
        if (Get-OwnedProcess $processNumber $processes.($kind + '_started')) { & taskkill.exe /PID $processNumber /T /F | Out-Null }
    }
    Write-Host 'YiGraph 已停止。'
}

function Get-OwnedProcess([int]$ProcessNumber, [long]$Started) {
    $process = Get-Process -Id $ProcessNumber -ErrorAction SilentlyContinue
    if ($process -and $process.StartTime.ToUniversalTime().Ticks -eq $Started) { return $process }
    return $null
}

try { Invoke-WindowsEntry; exit 0 }
catch { Write-Host ("未完成：" + $_.Exception.Message) -ForegroundColor Red; exit 1 }
