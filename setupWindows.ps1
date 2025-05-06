param (
    [string]$envName = "env",            # env name (default: env)
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$extraArgs                 # additional argument for `python -m venv`
)

# Path for activate script (Windows)
$activatePath = ".\$envName\Scripts\Activate.ps1"
$envAlreadyExists = Test-Path $envName

if ($envAlreadyExists) {
    Write-Host "Activating environment '$envName'..."
} else {
    Write-Host "Environment '$envName' does not exist yet. Creating environment..."
    $argsString = $extraArgs -join ' '
    $command = "python -m venv $envName $argsString"
    Write-Host "Exec: $command"
    Invoke-Expression $command
    if (-Not $?) {
        Write-Host "Error while creating the environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Environment was create successfully."
}

# Activate env
Write-Host "Activating environment"
& $activatePath

# Install requirements only the first time
if (-Not $envAlreadyExists) {
    if (Test-Path "requirements.txt") {
        Write-Host "Installing packets from requirements.txt..."
        pip install -r requirements.txt
        if ($?) {
            Write-Host "Packets successfully installed"
        } else {
            Write-Host "Error while installing requirements" -ForegroundColor Yellow
        }
    } else {
        Write-Host "requirements.txt not found" -ForegroundColor Yellow
    }
}