$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot
$python = 'E:\CondaEnvs\rofee-bearing\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Conda Python not found: $python"
}
Write-Output "Regression Python: $python"
& $python -m unittest tests.test_signal_transforms tests.test_plot_registry tests.test_signal_workbench tests.test_home_features tests.test_recognition_background tests.test_recognition_workbench tests.test_bearing_preview tests.test_bearing_ui_acceptance tests.test_spectrum_integration tests.test_spectrogram_integration tests.test_constellation_integration tests.test_feature_views tests.test_tsne_background tests.test_wavelet_views tests.test_bispectrum_views tests.test_jr_views tests.test_hht_views tests.test_hht_real_backend tests.test_uav_input tests.test_uav_contract tests.test_uav_intake_workflow tests.test_uav_background tests.test_uav_plot_panel tests.test_uav_main_integration tests.test_historical_generation tests.test_generation_background tests.test_historical_comparison_input tests.test_historical_model_comparison tests.test_historical_comparison_background tests.test_runtime_resources -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m unittest tests.test_waterfall tests.test_waterfall_migration -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python tests/smoke_test.py --models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$compileTargets = @(
    'airwatch',
    'models',
    'training',
    'tests',
    'tools',
    'packaging',
    'scripts'
)
$compileTargets += @(
    Get-ChildItem -LiteralPath $ProjectRoot -File -Filter '*.py' |
        ForEach-Object { $_.FullName }
)
& $python -m compileall -q @compileTargets
exit $LASTEXITCODE

