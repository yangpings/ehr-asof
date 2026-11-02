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


def test_train_writes_a_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", "--out", str(tmp_path / "data"), "--patients", "200"]) == 0
    out = tmp_path / "risk.pkl"
    assert main(["train", "--data", str(tmp_path / "data"), "--out", str(out)]) == 0
    assert out.exists()
    assert "trained risk model" in capsys.readouterr().out
