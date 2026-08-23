"""Four full-budget runs on one A100."""

from pathlib import Path
import shutil

from export_checkpoint import export_checkpoint
from tmp.oel.config import A100_RUNS
from tmp.oel.diagnostics import write_comparison_report
from tmp.oel.training import ROOT, run_accel, run_sfl


def upload(task, run_dir: Path) -> None:
    checkpoint = ROOT / "checkpoints" / run_dir.name / "0"
    export_checkpoint(run_dir, checkpoint)
    task.upload_artifact(run_dir.name, artifact_object=str(run_dir), wait_on_upload=True)
    task.upload_artifact(f"{run_dir.name}-checkpoint", artifact_object=str(checkpoint), wait_on_upload=True)
    print({"disk_free_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 1)}, flush=True)


def main() -> None:
    from clearml import Task

    task = Task.init(
        project_name="lukmanov-team/Open-Ended Learning",
        task_name="ued-four-methods-seed0",
        reuse_last_task_id=False,
        output_uri="s3://api.blackhole2.ai.innopolis.university:443/lukmanov-team",
    )
    task.connect({"runs": [config["name"] for config in A100_RUNS]})
    print({"disk_free_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 1)}, flush=True)
    run_dirs = []
    for config in A100_RUNS:
        run_dir = run_sfl(config) if config["method"] == "sfl" else run_accel(config)
        upload(task, run_dir)
        run_dirs.append(run_dir)
    comparison = ROOT / "runs" / "a100_comparison_seed0.html"
    write_comparison_report(run_dirs, comparison)
    task.upload_artifact("comparison", artifact_object=str(comparison), wait_on_upload=True)
    task.close()


if __name__ == "__main__":
    main()
