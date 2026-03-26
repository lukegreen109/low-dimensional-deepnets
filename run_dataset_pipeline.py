import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch as th

from compute_dist import compute_distance, join, join_didx, project
from get_geodesic import main as make_geodesic
from reparameterization import compute_lambda
from utils.configure import get_configs


def _abspath(repo_root: Path, p: str | None) -> str | None:
    if p is None:
        return None
    p = os.path.expanduser(str(p))
    if os.path.isabs(p):
        return p
    return str((repo_root / p).resolve())


def _first_existing_glob(patterns: list[str]) -> list[str]:
    for pat in patterns:
        fs = sorted(glob.glob(pat))
        if fs:
            return fs
    return []


def _extract_json_config(path: str) -> dict | None:
    """Extract the first JSON object embedded in a filename/path, or None."""
    try:
        s = path
        i0 = s.find("{")
        i1 = s.find("}")
        if i0 < 0 or i1 < 0 or i1 <= i0:
            return None
        return json.loads(s[i0 : i1 + 1])
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compute_lambda + InPCA distance/projection pipeline from a YAML config")
    parser.add_argument("--config", required=True, help="Path to dataset run config YAML (e.g. configs/runs/cifar10.yaml)")
    parser.add_argument("--force-lambda", action="store_true", help="Recompute and overwrite reindexed runs")
    parser.add_argument("--force-dist", action="store_true", help="Recompute and overwrite cached distance/join artifacts")
    parser.add_argument("--force-project", action="store_true", help="Recompute and overwrite cached per-seed projections")
    parser.add_argument("--skip-lambda", action="store_true", help="Skip compute_lambda step")
    parser.add_argument("--skip-dist", action="store_true", help="Skip pairwise distance + join step")
    parser.add_argument("--skip-project", action="store_true", help="Skip InPCA projection step")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = get_configs(args.config)

    labels_path = _abspath(repo_root, cfg.get("labels_path"))
    if not labels_path:
        raise ValueError("Config must set labels_path")
    labels = th.load(labels_path, weights_only=False)

    num_classes = int(cfg.get("num_classes"))

    results_root = _abspath(repo_root, cfg.get("save_loc"))
    if not results_root:
        raise ValueError("Config must set save_loc")
    inpca_root = _abspath(repo_root, cfg.get("inpca_save_loc"))
    if not inpca_root:
        raise ValueError("Config must set inpca_save_loc")

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(inpca_root, exist_ok=True)

    reindexed_dir = _abspath(repo_root, cfg.get("reindexed_dir"))
    if not reindexed_dir:
        reindexed_dir = str(Path(results_root) / "models" / "reindexed_new")
    geodesic_dir = _abspath(repo_root, cfg.get("geodesic_dir"))
    if not geodesic_dir:
        geodesic_dir = str(Path(results_root) / "models" / "loaded")

    os.makedirs(reindexed_dir, exist_ok=True)
    os.makedirs(geodesic_dir, exist_ok=True)

    runs_glob = cfg.get("runs_glob")
    models_dir = cfg.get("models_dir")
    run_files: list[str] = []
    if runs_glob:
        run_files = sorted(glob.glob(os.path.expanduser(runs_glob)))

    # If runs_glob is missing or empty, fall back to scanning models_dir.
    if not run_files:
        if not models_dir:
            raise ValueError("Config must set runs_glob or models_dir")
        md = os.path.expanduser(models_dir)
        run_files = _first_existing_glob(
            [
                os.path.join(md, "models", "all", "*.p"),
                os.path.join(md, "results", "models", "all", "*.p"),
                os.path.join(md, "models", "*.p"),
                os.path.join(md, "*.p"),
            ]
        )

    if not run_files:
        raise RuntimeError("No run files found (check runs_glob/models_dir in config)")

    didx_fn = cfg.get("lambda_didx_fn", "geod_all_progress")

    if not args.skip_lambda:
        compute_lambda(
            file_list=run_files,
            force=bool(args.force_lambda or cfg.get("force_lambda", False)),
            didx_loc=inpca_root,
            didx_fn=didx_fn,
            save_loc=reindexed_dir,
            labels_override=labels,
            nclasses=num_classes,
        )

    # Ensure a geodesic run exists
    geo_files = glob.glob(os.path.join(geodesic_dir, "*.p"))
    if not any("geodesic" in os.path.basename(f) for f in geo_files):
        make_geodesic(
            loc=geodesic_dir,
            name="",
            n=int(cfg.get("geodesic_n", 100)),
            ts=None,
            loaded=True,
            log=False,
            labels_override=labels,
            endpoint_labels_override=None,
            nclasses=num_classes,
        )

    groupby = cfg.get("GROUPBY", ["bsel"])
    idx_cols_cfg = cfg.get(
        "IDX_COLS",
        ["seed", "m", "opt", "t", "err", "verr", "bs", "aug", "lr", "wd", "bsel"],
    )
    # Normalize column naming: accept 'model' in configs but use 'm' internally.
    idx_cols = ["m" if c == "model" else c for c in idx_cols_cfg]
    if "m" not in idx_cols:
        idx_cols.append("m")

    join_fn = cfg.get("join_fn", "all_geod")
    keys = cfg.get("keys", ["yh", "yvh"])

    load_list = None
    if not args.skip_dist:
        allowed_basenames = {os.path.basename(p) for p in run_files}
        distance_sources = cfg.get("distance_sources")
        if distance_sources is None:
            run_dir = os.path.dirname(run_files[0]) if run_files else None
            sources = [reindexed_dir, geodesic_dir]
            if run_dir and run_dir not in sources:
                sources.append(run_dir)
        else:
            sources = [_abspath(repo_root, s) for s in distance_sources]

        fs: list[str] = []
        for d in sources:
            if d and os.path.isdir(d):
                fs.extend(glob.glob(os.path.join(d, "*.p")))

        # Avoid accidentally mixing in stale cached runs from previous experiments.
        # Keep only (a) current run basenames, plus (b) geodesic runs.
        filtered: list[str] = []
        for p in fs:
            if _extract_json_config(p) is None:
                continue
            base = os.path.basename(p)
            if "geodesic" in base or base in allowed_basenames:
                filtered.append(p)
        fs = sorted(set(filtered))
        if len(fs) < 2:
            preview = "\n".join(f" - {os.path.basename(p)}" for p in fs[:10])
            raise RuntimeError(
                "Need at least 2 runs with parseable JSON configs in their filenames "
                "(geodesic + at least one other run). Found %d.\n%s" % (len(fs), preview)
            )

        load_list = compute_distance(
            all_files=fs,
            groupby=groupby,
            save_didx=True,
            distf=str(cfg.get("distf", "dbhat")),
            save_loc=inpca_root,
            idx=idx_cols,
            parallel=int(cfg.get("parallel", 0)),
            force=bool(args.force_dist or cfg.get("force_dist", False)),
        )

        for key in keys:
            join_didx(
                loc=inpca_root,
                key=key,
                fn=join_fn,
                groupby=groupby,
                remove_f=False,
                load_list=load_list,
            )

            join(
                loc=inpca_root,
                key=key,
                fn=join_fn,
                groupby=groupby,
                save_loc=inpca_root,
                remove_f=False,
            )

            # project() expects didx_{key}_{join_fn}.p (because it reads didx_{fn}.p and fn includes key)
            src = os.path.join(inpca_root, f"didx_{join_fn}.p")
            dst = os.path.join(inpca_root, f"didx_{key}_{join_fn}.p")
            if os.path.exists(src) and (bool(args.force_dist or args.force_project) or not os.path.exists(dst)):
                shutil.copy2(src, dst)

    if not args.skip_project:
        didx_path = os.path.join(inpca_root, f"didx_{join_fn}.p")
        if not os.path.exists(didx_path):
            raise RuntimeError(f"Missing {didx_path}; run without --skip-dist first")

        df = th.load(didx_path, weights_only=False)
        try:
            df = df.reset_index(drop=True)
        except Exception:
            pass

        if "m" not in df.columns or "seed" not in df.columns:
            raise RuntimeError("didx table missing required columns 'm' and/or 'seed' (check IDX_COLS)")

        seeds = sorted(df.loc[df["m"] != "geodesic", "seed"].unique())
        if not seeds:
            seeds = [0]

        err_threshold = float(cfg.get("err_threshold", 1.0))
        for key in keys:
            for s in seeds:
                project(
                    seed=int(s),
                    fn=f"{key}_{join_fn}",
                    err_threshold=err_threshold,
                    extra_points=None,
                    loc=inpca_root,
                    force=bool(args.force_project or cfg.get("force_project", False)),
                )


if __name__ == "__main__":
    main()
