[CmdletBinding()]
param(
    [string]$Root = (Join-Path (Get-Location) 'YiGraph-dev'),
    [string]$Source,
    [string]$ConfigFile,
    [string]$Ref = 'main',
    [string]$IndexUrl = 'https://pypi.org/simple',
    [ValidateSet('Core', 'Engine', 'Web')][string]$Profile = 'Web',
    [switch]$WithPyg,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Invoke-YiGraphSetup {
    $script:InstallRoot = [IO.Path]::GetFullPath($Root)
    if ($Plan) {
        Write-Output "Profile: $Profile; install directory: $InstallRoot"
        Write-Output '1. Download portable uv and Git if needed; clone YiGraph or reuse -Source.'
        Write-Output '2. Install managed Python 3.11 and editable source in an isolated virtual environment.'
        Write-Output '3. Initialize workspace; verify actual graph demo.'
        if ($Profile -ne 'Core') { Write-Output '4. Install CPU engine dependencies, JDK 11 and Neo4j 3.5.25; verify import and database query.' }
        if ($Profile -eq 'Web') { Write-Output '5. Install Node.js 22.16.0 and npm dependencies; build frontend.' }
        if ($WithPyg) { Write-Output 'Optional: install CPU torch-geometric.' }
        Write-Output 'Write launchers and a verification report. LLM, embedding weights and Milvus require separate configuration/verification.'
        return
    }
    Assert-Platform
    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
    foreach ($folder in @('tools', 'cache', 'logs')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $InstallRoot $folder) | Out-Null
    }
    $script:Report = [ordered]@{ profile=$Profile; status='installing'; source=$null; commit=$null; demo_passed=$false; engine_import_passed=$false; neo4j_query_passed=$false; frontend_built=$false; pending=@('LLM service and credentials', 'Embedding model download and inference', 'Milvus service for text RAG'); error=$null }
    $log = Join-Path $InstallRoot ('logs/setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    Start-Transcript -Path $log | Out-Null
    try {
        Use-SystemProxy
        $uv = Install-Uv
        $checkout = Get-Checkout
        $Report.source = $checkout
        $Report.commit = 'source archive'
        if ($script:Git -and (Test-Path (Join-Path $checkout '.git'))) {
            $Report.commit = ((& $script:Git -C $checkout rev-parse HEAD) | Out-String).Trim()
        }
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallRoot 'tools/python'
        $env:UV_PYTHON_BIN_DIR = Join-Path $InstallRoot 'tools/python-bin'
        $env:UV_CACHE_DIR = Join-Path $InstallRoot 'cache/uv'
        $env:UV_NATIVE_TLS = 'true'
        $env:UV_INDEX_URL = $IndexUrl
        $env:UV_HTTP_TIMEOUT = '120'
        $env:UV_CONCURRENT_DOWNLOADS = '4'
        $env:PYTHONUTF8 = '1'
        $pythonArchive = '20251014/cpython-3.11.14+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz'
        Get-Archive ('https://github.com/astral-sh/python-build-standalone/releases/download/' + $pythonArchive.Replace('+', '%2B')) ('python/' + $pythonArchive) 'c2a7b0bb86eb9f1cc094a01bdabef7ddc77f89f8e45161fa7819f2b4a7ba7bc1' ''
        $pythonMirror = ([Uri]((Join-Path $InstallRoot 'cache/python') + '/')).AbsoluteUri
        Run $uv @('python', 'install', '3.11.14', '--mirror', $pythonMirror, '--no-registry')
        $venv = Join-Path $InstallRoot '.venv'
        $python = Join-Path $venv 'Scripts/python.exe'
        if (-not (Test-Path $python)) { Run $uv @('venv', '--python', '3.11.14', '--managed-python', $venv) }
        $env:PATH = (Join-Path $venv 'Scripts') + ';' + $env:PATH
        if ($Profile -ne 'Core' -or $WithPyg) {
            Run $uv @('pip', 'install', '--python', $python, 'torch==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cpu')
        }
        $package = $checkout
        $extras = @()
        if ($Profile -ne 'Core') { $extras += 'engine' }
        if ($Profile -eq 'Web') { $extras += 'web' }
        if ($WithPyg) { $extras += 'pyg' }
        if ($extras.Count) { $package += '[' + ($extras -join ',') + ']' }
        Run $uv @('pip', 'install', '--python', $python, '-e', $package)
        $env:YIGRAPH_HOME = Join-Path $InstallRoot 'workspace'
        Run (Join-Path $venv 'Scripts/yigraph.exe') @('init', $env:YIGRAPH_HOME)
        Run $python @((Join-Path $PSScriptRoot 'windows-runtime.py'), 'demo')
        $Report.demo_passed = $true
        $paths = @((Join-Path $venv 'Scripts'))
        if ($script:Git) { $paths += Split-Path $script:Git }
        $javaHome = ''
        $neoHome = ''
        if ($Profile -ne 'Core') {
            Run $python @((Join-Path $PSScriptRoot 'windows-runtime.py'), 'engine')
            $Report.engine_import_passed = $true
            $javaHome = Install-Jdk
            $neoHome = Install-Neo4j
            $env:JAVA_HOME = $javaHome
            $paths += Join-Path $javaHome 'bin'
            $env:PATH = (Join-Path $javaHome 'bin') + ';' + $env:PATH
            Run $python @((Join-Path $PSScriptRoot 'windows-runtime.py'), 'configure', '--neo-home', $neoHome)
            Test-Neo4j $python $neoHome
            $Report.neo4j_query_passed = $true
        }
        if ($ConfigFile) {
            Run $python @((Join-Path $PSScriptRoot 'windows-runtime.py'), 'apply-config', '--config', $ConfigFile, '--source', $checkout)
        }
        if ($Profile -eq 'Web') {
            $nodeHome = Install-Node
            $paths += $nodeHome
            $env:PATH = $nodeHome + ';' + $env:PATH
            Push-Location $checkout
            try {
                $npm = Join-Path $nodeHome 'npm.cmd'
                $npmAction = 'install'
                if (Test-Path 'package-lock.json') { $npmAction = 'ci' }
                Run $npm @($npmAction, '--ignore-scripts')
                $frontend = Join-Path $checkout 'web/frontend/next-app'
                $npmAction = 'install'
                if (Test-Path (Join-Path $frontend 'package-lock.json')) { $npmAction = 'ci' }
                Run $npm @($npmAction, '--prefix', $frontend)
                Run $npm @('run', 'build')
                if (-not (Test-Path (Join-Path $frontend 'out/index.html'))) { throw 'Frontend build did not produce out/index.html.' }
                $Report.frontend_built = $true
            } finally { Pop-Location }
        }
        Run $uv @('pip', 'check', '--python', $python)
        $env:HF_HOME = Join-Path $InstallRoot 'cache/huggingface'
        Write-Activation $paths $javaHome
        @{python=$python;neo4j=$neoHome;source=$checkout} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $InstallRoot 'installation.json')
        $Report.status = 'environment_ready'
        if ($Profile -eq 'Core') { $Report.status = 'ready'; $Report.pending = @() }
        Write-Output "Environment installed. Verification report: $InstallRoot/install-report.json"
    } catch {
        $Report.status = 'failed'
        $Report.error = $_.Exception.Message
        throw
    } finally {
        $Report | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $InstallRoot 'install-report.json')
        Stop-Transcript | Out-Null
    }
}

function Assert-Platform {
    if ($env:OS -ne 'Windows_NT' -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'This installer supports Windows x64 only.' }
    if ($WithPyg -and $Profile -eq 'Core') { Write-Output 'Core + WithPyg also installs CPU PyTorch.' }
}

function Run([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program" }
}

function Use-SystemProxy {
    if (-not $env:HTTPS_PROXY) {
        $target = [Uri]'https://pypi.org'
        $proxy = [Net.WebRequest]::DefaultWebProxy.GetProxy($target)
        if ($proxy -and $proxy.Authority -ne $target.Authority) {
            $env:HTTPS_PROXY = $proxy.AbsoluteUri
            Write-Output 'Using the configured Windows system proxy for this installer process.'
        }
    }
}

function Get-Archive([string]$Uri, [string]$Name, [string]$Sha256, [string]$Destination) {
    $archive = Join-Path $InstallRoot ('cache/' + $Name)
    New-Item -ItemType Directory -Force -Path (Split-Path $archive) | Out-Null
    if (-not (Test-Path $archive)) {
        Write-Output "Downloading $Name"
        $curl = Get-Command (Join-Path $env:SystemRoot 'System32/curl.exe') -ErrorAction SilentlyContinue
        if ($curl) {
            Run $curl.Source @('--fail', '--location', '--silent', '--show-error', '--retry', '2', '--connect-timeout', '20', '--max-time', '900', '--output', ($archive + '.part'), $Uri)
        } else {
            $client = New-Object Net.WebClient
            try { $client.DownloadFile($Uri, ($archive + '.part')) } finally { $client.Dispose() }
        }
        if ((Get-FileHash ($archive + '.part') -Algorithm SHA256).Hash -ine $Sha256) { throw "Checksum mismatch: $Name" }
        Move-Item -LiteralPath ($archive + '.part') -Destination $archive -Force
    }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ine $Sha256) { throw "Cached checksum mismatch: $archive" }
    if ($Destination) { Expand-Archive -LiteralPath $archive -DestinationPath $Destination -Force }
}

function Install-Uv {
    $destination = Join-Path $InstallRoot 'tools/uv'
    $uv = Join-Path $destination 'uv.exe'
    if (-not (Test-Path $uv)) {
        Get-Archive 'https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-pc-windows-msvc.zip' 'uv-0.9.5.zip' '515dc53d7553f1357d0abc1f70acd921fbb9e30230b1d9a08737236daa6ee920' $destination | Out-Host
    }
    return $uv
}

function Get-Checkout {
    if ($Source) {
        $checkout = [IO.Path]::GetFullPath($Source)
        if (-not (Test-Path (Join-Path $checkout 'pyproject.toml'))) { throw "YiGraph source directory not found: $checkout" }
        $gitCommand = Get-Command git -ErrorAction SilentlyContinue
        if ($gitCommand) { $script:Git = $gitCommand.Source }
        return $checkout
    }
    $gitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCommand) { $script:Git = $gitCommand.Source } else {
        $script:Git = Join-Path $InstallRoot 'tools/git/cmd/git.exe'
        if (-not (Test-Path $script:Git)) {
            $release = Invoke-RestMethod 'https://api.github.com/repos/git-for-windows/git/releases/latest'
            $asset = $release.assets | Where-Object { $_.name -match '^MinGit-[0-9].*-64-bit.zip$' } | Select-Object -First 1
            if (-not $asset -or $asset.digest -notmatch '^sha256:[a-f0-9]{64}$') { throw 'No checksum-bearing Windows x64 MinGit release found.' }
            Get-Archive $asset.browser_download_url $asset.name $asset.digest.Substring(7) (Join-Path $InstallRoot 'tools/git') | Out-Host
        }
    }
    $checkout = Join-Path $InstallRoot 'source'
    if ($Source) { $checkout = [IO.Path]::GetFullPath($Source) }
    if (-not (Test-Path $checkout)) {
        if ($Source) { throw "Source directory does not exist: $checkout" }
        Run $script:Git @('clone', '--depth', '1', '--branch', $Ref, 'https://github.com/iDC-NEU/YiGraph.git', $checkout) | Out-Host
    }
    if (-not (Test-Path (Join-Path $checkout 'pyproject.toml')) -or -not (Test-Path (Join-Path $checkout '.git'))) { throw 'Source must be a YiGraph Git checkout with pyproject.toml.' }
    return $checkout
}

function Install-Jdk {
    $destination = Join-Path $InstallRoot 'tools/jdk'
    $java = Get-ChildItem $destination -Filter java.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $java) {
        $assets = Invoke-RestMethod 'https://api.adoptium.net/v3/assets/latest/11/hotspot?architecture=x64&image_type=jdk&os=windows&vendor=eclipse'
        $package = $assets[0].binary.package
        Get-Archive $package.link $package.name $package.checksum $destination | Out-Host
        $java = Get-ChildItem $destination -Filter java.exe -Recurse | Select-Object -First 1
    }
    return $java.Directory.Parent.FullName
}

function Install-Neo4j {
    $destination = Join-Path $InstallRoot 'tools/neo4j'
    $neoHome = Join-Path $destination 'neo4j-community-3.5.25'
    if (-not (Test-Path (Join-Path $neoHome 'bin/neo4j.bat'))) {
        $uri = 'https://dist.neo4j.org/neo4j-community-3.5.25-windows.zip'
        $checksum = (Get-RemoteText ($uri + '.sha256')).Trim().Split(' ')[0]
        Get-Archive $uri 'neo4j-community-3.5.25-windows.zip' $checksum $destination | Out-Host
    }
    return $neoHome
}

function Install-Node {
    $name = 'node-v22.16.0-win-x64'
    $destination = Join-Path $InstallRoot 'tools/node'
    $nodeHome = Join-Path $destination $name
    if (-not (Test-Path (Join-Path $nodeHome 'node.exe'))) {
        $base = 'https://nodejs.org/dist/v22.16.0/'
        $sums = Get-RemoteText ($base + 'SHASUMS256.txt')
        $line = ($sums -split "`n" | Where-Object { $_.Trim().EndsWith(" $name.zip") })
        if (-not $line) { throw 'Node archive checksum not found.' }
        Get-Archive ($base + $name + '.zip') ($name + '.zip') ($line.Trim() -split '\s+')[0] $destination | Out-Host
    }
    return $nodeHome
}

function Test-Neo4j([string]$Python, [string]$NeoHome) {
    foreach ($port in @(7474,7687)) {
        $socket = New-Object Net.Sockets.TcpClient
        try { $socket.Connect('127.0.0.1', $port); throw "Port $port is already in use. Stop the conflicting service before retrying." }
        catch [Net.Sockets.SocketException] { } finally { $socket.Dispose() }
    }
    $escapedHome = $NeoHome.Replace("'", "''")
    $command = "& '$escapedHome/bin/neo4j.bat' console"
    $stdout = Join-Path $InstallRoot 'logs/neo4j.stdout.log'
    $stderr = Join-Path $InstallRoot 'logs/neo4j.stderr.log'
    $process = Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command', $command) -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    try {
        Run $Python @((Join-Path $PSScriptRoot 'windows-runtime.py'), 'neo4j')
    } finally {
        # Only stop the process tree created for this verification.
        if (-not $process.HasExited) { & taskkill.exe /PID $process.Id /T /F | Out-Null }
    }
}

function Get-RemoteText([string]$Uri) {
    $client = New-Object Net.WebClient
    try { return $client.DownloadString($Uri) } finally { $client.Dispose() }
}

function Write-Activation([string[]]$Paths, [string]$JavaHome) {
    $escapedRoot = $InstallRoot.Replace("'", "''")
    $activation = "`$env:YIGRAPH_HOME = '$escapedRoot/workspace'`r`n"
    if ($JavaHome) { $activation += "`$env:JAVA_HOME = '" + $JavaHome.Replace("'", "''") + "'`r`n" }
    $activation += "`$env:PATH = '" + (($Paths -join ';').Replace("'", "''")) + ";' + `$env:PATH`r`n"
    if ($env:HTTPS_PROXY) { $activation += "`$env:HTTPS_PROXY = '" + $env:HTTPS_PROXY.Replace("'", "''") + "'`r`n" }
    $activation += "`$env:PYTHONUTF8 = '1'`r`n`$env:PYTHONUNBUFFERED = '1'`r`n"
    $activation += "`$env:HF_HOME = '$escapedRoot/cache/huggingface'`r`n"
    Set-Content -Encoding UTF8 (Join-Path $InstallRoot 'activate.ps1') $activation
}

Invoke-YiGraphSetup
