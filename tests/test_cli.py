from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_dlp_guard.cli import main

TEXT = "Mail jane.doe@example.org, key AKIAIOSFODNN7ABCDEFG, ip 8.8.8.8\n"


@pytest.fixture()
def sample(tmp_path: Path) -> Path:
    p = tmp_path / "in.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p


def test_scan(sample: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan", str(sample)]) == 0
    out = capsys.readouterr().out
    assert "EMAIL" in out and "AWS_ACCESS_KEY" in out and "3 findings" in out
    assert "AKIAIOSFODNN7ABCDEFG" not in out
    assert main(["scan", str(sample), "--show-values"]) == 0
    assert "AKIAIOSFODNN7ABCDEFG" in capsys.readouterr().out
    assert main(["scan", str(sample), "--json", "--backend", "python"]) == 0
    assert [f["label"] for f in json.loads(capsys.readouterr().out)] == [
        "EMAIL",
        "AWS_ACCESS_KEY",
        "IP_ADDRESS",
    ]


def test_redact_and_block(sample: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["redact", str(sample)]) == 0
    captured = capsys.readouterr()
    assert "[AWS_ACCESS_KEY]" in captured.out and "jane.doe@example.org" not in captured.out
    assert "EMAIL:tokenize x1" in captured.err
    assert main(["redact", str(sample), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "redact"
    policy = tmp_path / "p.yaml"
    policy.write_text("rules:\n  - labels: [EMAIL]\n    action: block\n", encoding="utf-8")
    assert main(["redact", str(sample), "--policy", str(policy)]) == 2
    assert "BLOCKED" in capsys.readouterr().err
    assert main(["redact", str(sample), "--policy", str(policy), "--json"]) == 2
    empty = tmp_path / "e.txt"
    empty.write_text("nothing", encoding="utf-8")
    assert main(["redact", str(empty)]) == 0
    assert "nothing found" in capsys.readouterr().err


def test_stream(sample: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stream", str(sample), "--chunk", "3", "--direction", "prompt"]) == 0
    assert "[AWS_ACCESS_KEY]" in capsys.readouterr().out
    policy = tmp_path / "p.yaml"
    policy.write_text("rules:\n  - labels: [IP_ADDRESS]\n    action: block\n", encoding="utf-8")
    assert main(["stream", str(sample), "--policy", str(policy)]) == 2


def test_gen_secrets_and_catalog(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "s.jsonl"
    assert main(["gen-secrets", "--positives", "5", "--negatives", "3", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 8 and rows[-1]["kind"] == "negative"
    assert main(["gen-secrets", "--positives", "1", "--negatives", "0"]) == 0
    assert '"spans"' in capsys.readouterr().out
    assert main(["catalog"]) == 0
    assert "private_key" in capsys.readouterr().out


def test_train_tagger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from .conftest import make_records

    texts, spans = make_records(40, seed=2)
    split = tmp_path / "data" / "train"
    split.mkdir(parents=True)
    with (split / "1en.jsonl").open("w", encoding="utf-8") as fh:
        for t, sp in zip(texts, spans, strict=True):
            label = {
                "GIVEN_NAME": "GIVENNAME",
                "SURNAME": "SURNAME",
                "CITY": "CITY",
                "STREET": "STREET",
                "BUILDING_NUMBER": "BUILDINGNUM",
            }
            mask = [
                {"start": s, "end": e, "label": label[lab], "value": t[s:e]} for s, e, lab in sp
            ]
            fh.write(json.dumps({"source_text": t, "privacy_mask": mask, "language": "en"}) + "\n")
    model = tmp_path / "m.crf"
    args = [
        "train-tagger",
        "--data",
        str(tmp_path / "data"),
        "--out",
        str(model),
        "--languages",
        "en",
        "--epochs",
        "1",
        "--bucket-bits",
        "12",
    ]
    assert main(args) == 0
    assert model.exists() and "saved" in capsys.readouterr().err
    probe = tmp_path / "probe.txt"
    probe.write_text(texts[0], encoding="utf-8")
    assert main(["scan", str(probe), "--model", str(model), "--json"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)
