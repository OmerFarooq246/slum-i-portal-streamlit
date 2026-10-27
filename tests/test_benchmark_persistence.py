"""Verify completed measurements survive an interrupted benchmark configuration."""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import system_benchmark as benchmark


@pytest.mark.parametrize("pipeline", ["streaming", "in_memory"])
def test_interruption_retains_raw_runs_outside_artifact_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline: str
) -> None:
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        model="test-model",
        pipeline=pipeline,
        results_dir=Path("shared/results"),
        work_dir=Path("local/artifacts"),
        cpu_limit=2,
        batch_size=1,
        warmup_runs=1,
        measured_runs=5,
        crop_x=0,
        crop_y=0,
    )
    adapter = SimpleNamespace(
        default_weights=Path("weights.h5"),
        supporting_artifacts=(),
        load=lambda _: object(),
        infer=lambda: None,
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "MODEL_ADAPTERS", {args.model: adapter})
    monkeypatch.setattr(benchmark, "ROI_SPECS", [("small", 1)])
    monkeypatch.setattr(benchmark, "validate_inputs", lambda *_: None)
    monkeypatch.setattr(benchmark, "path_sha256", lambda _: "fixture-hash")
    for target, name in (
        (benchmark.tf.config.threading, "set_intra_op_parallelism_threads"),
        (benchmark.tf.config.threading, "set_inter_op_parallelism_threads"),
        (benchmark.tf.config.experimental, "set_synchronous_execution"),
    ):
        monkeypatch.setattr(target, name, lambda _: None)

    calls = 0

    def interrupt(*positional):
        nonlocal calls
        calls += 1
        assert positional[7] == args.work_dir
        # One warm-up, two completed measurements, then interruption.
        if calls == 4:
            raise RuntimeError("simulated interruption")
        return {"total_latency_ms": float(calls)}

    monkeypatch.setattr(benchmark, "run_pipeline", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        benchmark.main()

    with (args.results_dir / "raw_runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run"] for row in rows] == ["1", "2"]
    assert [row["total_latency_ms"] for row in rows] == ["2.0", "3.0"]
    assert all(row["pipeline"] == pipeline for row in rows)
    assert not (args.results_dir / "summary.csv").exists()
    assert not (args.results_dir / "paper_table.md").exists()
