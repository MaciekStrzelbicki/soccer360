"""CLI interface for Soccer360 pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger("soccer360.cli")


def _load_cli_utils():
    """Import CLI runtime helpers lazily so `--help` works in thin envs."""
    try:
        from .utils import load_config, setup_logging
    except ModuleNotFoundError as exc:
        if exc.name == "yaml":
            raise click.ClickException(
                "PyYAML is required to run Soccer360 commands. "
                "Install project dependencies or use the Docker workflow."
            )
        raise
    return load_config, setup_logging


@click.group()
@click.option(
    "--config", "-c",
    default="configs/pipeline.yaml",
    envvar="SOCCER360_CONFIG",
    help="Path to pipeline config YAML.",
)
@click.pass_context
def cli(ctx: click.Context, config: str):
    """Soccer360 -- Automated 360 soccer video processing."""
    load_config, setup_logging = _load_cli_utils()
    ctx.ensure_object(dict)
    cfg = load_config(config)

    paths = cfg.setdefault("paths", {})
    paths.setdefault("ingest", "/tank/ingest")
    paths.setdefault("scratch", "/scratch/work")
    paths.setdefault("processed", "/tank/processed")
    paths.setdefault("highlights", "/tank/highlights")
    paths.setdefault("models", "/tank/models")
    paths.setdefault("labeling", "/tank/labeling")
    paths.setdefault("stagging", "/tank/stagging")
    paths.setdefault("archive_raw", "/tank/archive_raw")
    paths.setdefault("logs", "/tank/logs")

    # Relative paths (local, non-Docker configs) -> absolute, so ffmpeg concat
    # lists and worker processes resolve them regardless of their cwd.
    for key, value in list(paths.items()):
        if isinstance(value, str) and not Path(value).is_absolute():
            paths[key] = str(Path(value).resolve())

    cal = cfg.get("calibration")
    if isinstance(cal, dict) and isinstance(cal.get("dir"), str) and not Path(cal["dir"]).is_absolute():
        cal["dir"] = str(Path(cal["dir"]).resolve())

    Path(paths["scratch"]).mkdir(parents=True, exist_ok=True)
    Path(paths["stagging"]).mkdir(parents=True, exist_ok=True)
    log_dir = Path(paths["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    paths["logs"] = str(log_dir)

    ctx.obj["config"] = cfg
    log_cfg = cfg.get("logging", {})
    log_name = Path(log_cfg.get("file") or "soccer360.log").name
    log_file = str(log_dir / log_name)
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_file)


@cli.command()
@click.pass_context
def watch(ctx: click.Context):
    """Start the watch-folder daemon on the ingest directory."""
    from .watcher import WatcherDaemon

    daemon = WatcherDaemon(ctx.obj["config"])
    daemon.run()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--no-cleanup", is_flag=True, help="Keep scratch files after processing.")
@click.pass_context
def process(ctx: click.Context, file_path: str, no_cleanup: bool):
    """Process a single 360 video file."""
    from .pipeline import Pipeline

    pipe = Pipeline(ctx.obj["config"])
    pipe.run(file_path, cleanup=not no_cleanup)


@cli.command()
@click.option("--epochs", default=50, help="Training epochs.")
@click.option("--data", required=True, type=click.Path(exists=True), help="YOLO dataset YAML path.")
@click.option("--base-model", default=None, type=click.Path(), help="Optional local base model .pt path.")
@click.option("--output-model-name", default=None, help="Optional promoted model filename (defaults to ball_best.pt).")
@click.option("--update-active/--no-update-active", default=True, help="Also copy the trained best weights to ball_best.pt for inference.")
@click.pass_context
def train(
    ctx: click.Context,
    epochs: int,
    data: str,
    base_model: str | None,
    output_model_name: str | None,
    update_active: bool,
):
    """Retrain / fine-tune the YOLO ball detection model."""
    from .trainer import Trainer

    trainer = Trainer(ctx.obj["config"])
    trainer.run(
        data=data,
        epochs=epochs,
        base_model=base_model,
        output_model_name=output_model_name,
        update_active=update_active,
    )


@cli.command("export-hard-frames")
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("detections_json", type=click.Path(exists=True))
@click.option("--threshold", default=0.3, help="Confidence threshold for hard frames.")
@click.option("--output-dir", default=None, help="Output directory (defaults to config labeling path).")
@click.pass_context
def export_hard_frames(
    ctx: click.Context, video_path: str, detections_json: str,
    threshold: float, output_dir: str | None,
):
    """Export low-confidence frames for labeling."""
    from .trainer import Trainer

    cfg = ctx.obj["config"]
    out = output_dir or cfg["paths"]["labeling"]
    trainer = Trainer(cfg)
    trainer.export_hard_frames(
        video_path=video_path,
        detections_path=Path(detections_json),
        threshold=threshold,
        output_dir=Path(out),
    )


@cli.command()
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--llm/--no-llm", default=None, help="Use the vision LLM (default: calibration.llm.enabled).")
@click.option("--port", default=8088, help="Dashboard port for the printed calibration URL.")
@click.pass_context
def calibrate(ctx: click.Context, video_path: str, llm: bool | None, port: int):
    """Find scoreboard and hoops in a 360 video (auto), save a preview.

    The result is a starting point for the dashboard calibration page,
    where it can be corrected and saved as the manual calibration.
    """
    import copy
    import json

    import cv2

    from .calibration import (
        calibration_config, grab_equirect, manual_path, preview_image,
        register_video, resolve_calibration,
    )
    from .utils import probe_video

    cfg = copy.deepcopy(ctx.obj["config"])
    cal_cfg = cfg.setdefault("calibration", {})
    cal_cfg["enabled"] = True
    if llm is not None:
        cal_cfg.setdefault("llm", {})["enabled"] = llm
    meta = probe_video(video_path)
    reg = register_video(cfg, video_path)

    # ignore an existing manual file: we want a fresh automatic result
    saved = json.loads(reg.read_text())
    reg.write_text(json.dumps({**saved, "scoreboard": None, "hoops": []}, indent=2))
    try:
        cal = resolve_calibration(cfg, video_path, meta.duration)
    finally:
        reg.write_text(json.dumps(saved, indent=2))

    auto_path = manual_path(cfg, video_path).with_suffix(".auto.json")
    auto_path.write_text(json.dumps(cal, indent=2))
    t = meta.duration * calibration_config(cfg)["sample_time_frac"]
    prev = preview_image(grab_equirect(video_path, t, 3840), cal)
    prev_path = auto_path.with_suffix(".png")
    cv2.imwrite(str(prev_path), prev[..., ::-1])

    click.echo(json.dumps(cal, indent=2))
    click.echo(f"Auto result: {auto_path}")
    click.echo(f"Preview:     {prev_path}")
    click.echo(f"Adjust/save: http://localhost:{port}/calibrate/{Path(video_path).stem}")


@cli.command()
@click.option("--port", default=8088, help="Dashboard server port.")
@click.option("--host", default="0.0.0.0", help="Dashboard server host.")
@click.pass_context
def dashboard(ctx: click.Context, port: int, host: str):
    """Start the monitoring dashboard web server."""
    import uvicorn

    from .dashboard import create_app

    app = create_app(ctx.obj["config"])
    logger.info("Starting Soccer360 Dashboard on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
