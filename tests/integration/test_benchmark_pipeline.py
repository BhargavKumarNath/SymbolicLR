import pytest
import sys
import subprocess
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_all_heavy_dependencies():
    """
    Mocks standard heavy computations so the integration test runs in < 2 seconds, while still executing structural logic of benchmark.py
    """
    with patch("data.fidelity.FidelityManager.get_low_fidelity") as mock_data, \
         patch("models.probe.ProbeTrainer.evaluate_schedule", return_value=1.0) as mock_train, \
         patch("gp.evaluator.symbolr_rust.evaluate_fast", return_value=MagicMock()) as mock_rust:
        
        # Return mock iterables for train/val loaders
        mock_data.return_value = (MagicMock(), MagicMock())
        yield
    
def test_benchmark_dry_run(mock_all_heavy_dependencies):
    """
    Executes benchmark.py programmatically with minimal parameters to ensure the entire pipeline is structurally sound
    """
    import benchmark

    # Simulate CLI arguments for a micro-run: 1 Gen, 4 Individuals, 2 Workers
    test_args =[
        "benchmark.py", 
        "--generations", "1", 
        "--pop_size", "4", 
        "--epochs", "1",
        "--workers", "2",
        "--seed", "42"
    ]
    
    with patch.object(sys, 'argv', test_args):
        try:
            benchmark.main()
        except Exception as e:
            pytest.fail(f"Benchmark pipeline threw an unexpected error: {e}")

def test_cli_help_command():
    """Validates the CLI parses arguments correctly via a subprocess call."""
    result = subprocess.run([sys.executable, "benchmark.py", "--help"], 
        capture_output=True, 
        text=True
    )
    assert result.returncode == 0
    assert "SymboLR" in result.stdout
    assert "--generations" in result.stdout
