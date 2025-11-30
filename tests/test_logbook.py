import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logbook


def test_log_run_creates_output_directory(tmp_path, monkeypatch):
    log_path = tmp_path / "outputs" / "logbook.csv"
    monkeypatch.setattr(logbook, "LOGBOOK_PATH", str(log_path))

    assert not os.path.exists(log_path)

    logbook.log_run(
        model="test_model",
        seed=123,
        params={"param": 1},
        output_csv="result.csv",
        segments="segment_info",
        notes="note",
    )

    assert os.path.isfile(log_path)
    with open(log_path, newline="") as file:
        rows = list(csv.reader(file))

    assert rows[0] == [
        "Date",
        "Model",
        "Seed",
        "Params",
        "Output CSV",
        "BB Segments",
        "Notes",
    ]
    assert rows[1][1:] == [
        "test_model",
        "123",
        "{'param': 1}",
        "result.csv",
        "segment_info",
        "note",
    ]
