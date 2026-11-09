# Cai dat phu thuoc va kiem tra moi truong. Chay tai thu muc DuAnBigData.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "Python:"
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_pipeline.py --mode check
if ($LASTEXITCODE -ne 0) {
    Write-Host "Check failed. Dat du 5 CSV vao data\ roi chay lai."
    exit $LASTEXITCODE
}
Write-Host "Tiep theo: python run_pipeline.py --reset --mode all --sample-chunks 2"
