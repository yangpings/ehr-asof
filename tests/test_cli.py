from __future__ import annotations

from pathlib import Path

import pytest

from ehr_asof.cli import main


def test_synth_then_delays(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", "--out", str(tmp_path), "--patients", "40"]) == 0
    assert (tmp_path / "hosp" / "labevents.csv.gz").exists()
    assert main(["delays", "--data", str(tmp_path)]) == 0
    assert "median_h" in capsys.readouterr().out


def test_missing_data_dir_exit_code(tmp_path: Path) -> None:
    assert main(["delays", "--data", str(tmp_path / "nope")]) == 2
