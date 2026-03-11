import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import inspect
import json
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = (ROOT_DIR / "outputs").resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from models.utils import solution_cost  # noqa: E402
except Exception:
    solution_cost = None

try:
    from benchmarking.generatorLarge import (
        generate_instance as generate_synthetic_instance,  # noqa: E402
    )
except Exception:
    generate_synthetic_instance = None


class ModelSpec:
    def __init__(self, model_name, module_name, function_name, function, source_file):
        self.model_name = model_name
        self.module_name = module_name
        self.function_name = function_name
        self.function = function
        self.source_file = source_file


def _slugify(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_")


def _instance_to_filename(instance_id):
    return _slugify(instance_id.replace("/", "__"))


def _is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sanitize_relative_parts(path_obj):
    parts = []
    for part in path_obj.parts:
        if part in ("", ".", ".."):
            continue
        parts.append(part)
    if parts and parts[0] == "outputs":
        parts = parts[1:]
    return parts


def _normalize_output_path(path_value, outputs_root=OUTPUTS_ROOT):
    """
    Force output paths under `outputs/`.
    Returns (normalized_path, was_rebased).
    """
    raw = Path(path_value)
    resolved = raw.resolve() if raw.is_absolute() else (ROOT_DIR / raw).resolve()
    if _is_relative_to(resolved, outputs_root):
        return resolved, False

    if raw.is_absolute():
        fallback_name = raw.name or "results"
        normalized = (outputs_root / fallback_name).resolve()
    else:
        safe_parts = _sanitize_relative_parts(raw)
        if not safe_parts:
            safe_parts = ["results"]
        normalized = (outputs_root.joinpath(*safe_parts)).resolve()

    return normalized, True


def _import_module_from_path(module_path, index):
    module_stem = _slugify(module_path.stem.replace("-", "_"))
    dynamic_module_name = f"benchmark_dynamic_{module_stem}_{index}"
    spec = importlib.util.spec_from_file_location(dynamic_module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dynamic_module_name


def _is_runnable_solver(fn):
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    params = [
        p
        for p in signature.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if not params:
        return False
    if params[0].name != "instance":
        return False

    required = [p for p in params if p.default is inspect._empty]
    return len(required) == 1


def _fallback_solver_functions(module):
    candidates = []
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__module__ != module.__name__:
            continue
        if name.startswith("_"):
            continue
        if not _is_runnable_solver(fn):
            continue
        candidates.append((name, fn))
    return sorted(candidates, key=lambda x: x[0])


def discover_models(models_dir=ROOT_DIR / "models"):
    """
    Discover solver functions from Python files in `models/`.

    Priority:
    1) `solve(instance)` if present.
    2) fallback to public callables whose only required argument is `instance`.
    """
    models_path = Path(models_dir).resolve()
    if not models_path.exists():
        raise FileNotFoundError(f"models directory not found: {models_path}")

    discovered = []

    if str(models_path.parent) not in sys.path:
        sys.path.insert(0, str(models_path.parent))
    if str(models_path) not in sys.path:
        sys.path.insert(0, str(models_path))

    py_files = sorted(
        p
        for p in models_path.glob("*.py")
        if not p.name.startswith("__") and p.stem not in {"utils"}
    )

    for idx, module_path in enumerate(py_files):
        try:
            module, module_name = _import_module_from_path(module_path, idx)
        except Exception as exc:
            print(f"[discover] skipping {module_path.name}: import failed ({exc})")
            continue

        solve_fn = getattr(module, "solve", None)
        if callable(solve_fn) and _is_runnable_solver(solve_fn):
            discovered.append(
                ModelSpec(
                    model_name=_slugify(module_path.stem),
                    module_name=module_name,
                    function_name="solve",
                    function=solve_fn,
                    source_file=module_path,
                )
            )
            continue

        fallback = _fallback_solver_functions(module)
        if not fallback:
            print(
                f"[discover] skipping {module_path.name}: no runnable solve-like function"
            )
            continue

        multi = len(fallback) > 1
        for fn_name, fn in fallback:
            if multi:
                model_name = _slugify(f"{module_path.stem}__{fn_name}")
            else:
                model_name = _slugify(module_path.stem)
            discovered.append(
                ModelSpec(
                    model_name=model_name,
                    module_name=module_name,
                    function_name=fn_name,
                    function=fn,
                    source_file=module_path,
                )
            )

    return discovered


def _coerce_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    for attr in ("item",):
        if hasattr(value, attr) and callable(getattr(value, attr)):
            try:
                return getattr(value, attr)()
            except Exception:
                pass

    return repr(value)


def _extract_solution_fields(solver_output, instance, elapsed_sec):
    routes = None
    total_distance = None
    reported_runtime_sec = None
    raw_payload = {}

    if isinstance(solver_output, dict):
        raw_payload = dict(solver_output)
        routes = _first_not_none(
            solver_output.get("routes"),
            solver_output.get("best_routes"),
            solver_output.get("solution"),
        )
        total_distance = _first_not_none(
            _coerce_float(solver_output.get("total_distance")),
            _coerce_float(solver_output.get("best_cost")),
            _coerce_float(solver_output.get("cost")),
            _coerce_float(solver_output.get("distance")),
            _coerce_float(solver_output.get("objective")),
            _coerce_float(solver_output.get("final_cost")),
        )
        reported_runtime_sec = _coerce_float(solver_output.get("runtime_sec"))
    else:
        raw_payload = {"solver_output": solver_output}
        if hasattr(solver_output, "routes"):
            routes = getattr(solver_output, "routes")
        if hasattr(solver_output, "total_distance"):
            total_distance = _coerce_float(getattr(solver_output, "total_distance"))
        if hasattr(solver_output, "runtime_sec"):
            reported_runtime_sec = _coerce_float(getattr(solver_output, "runtime_sec"))
        if routes is None and isinstance(solver_output, (list, tuple)):
            routes = solver_output

    if total_distance is None and routes is not None and solution_cost is not None:
        try:
            total_distance = float(solution_cost(instance, routes))
        except Exception:
            total_distance = None

    return routes, total_distance, elapsed_sec, reported_runtime_sec, raw_payload


def _run_single_model_instance(model, instance_id, instance, result_path):
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    try:
        solver_output = model.function(instance)
        elapsed = time.perf_counter() - t0
        (
            routes,
            total_distance,
            runtime_sec,
            reported_runtime_sec,
            raw_payload,
        ) = _extract_solution_fields(solver_output, instance, elapsed)

        row = {
            "status": "ok",
            "error": None,
            "model_name": model.model_name,
            "module_name": model.module_name,
            "function_name": model.function_name,
            "source_file": str(model.source_file),
            "instance_id": instance_id,
            "started_at_utc": started.isoformat(),
            "runtime_sec": runtime_sec,
            "reported_runtime_sec": reported_runtime_sec,
            "total_distance": total_distance,
            "routes": _to_jsonable(routes),
            "raw_output": _to_jsonable(raw_payload),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        row = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "model_name": model.model_name,
            "module_name": model.module_name,
            "function_name": model.function_name,
            "source_file": str(model.source_file),
            "instance_id": instance_id,
            "started_at_utc": started.isoformat(),
            "runtime_sec": elapsed,
            "reported_runtime_sec": None,
            "total_distance": None,
            "routes": None,
            "raw_output": None,
        }

    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_models(
    models,
    instances,
    outputs_dir=ROOT_DIR / "outputs",
    overwrite=False,
    stop_on_error=False,
    jobs=1,
):
    """
    Execute discovered models on all instances and write JSON outputs.

    Output path format:
      outputs/{model_name}/{instance}.json
    """
    out_root = Path(outputs_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    jobs = max(1, int(jobs))
    tasks = []

    for model in models:
        model_dir = out_root / model.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for instance_id, instance in instances.items():
            filename = f"{_instance_to_filename(instance_id)}.json"
            result_path = model_dir / filename

            if result_path.exists() and not overwrite:
                continue
            tasks.append((model, instance_id, instance, result_path))

    if jobs == 1:
        for model, instance_id, instance, result_path in tasks:
            row = _run_single_model_instance(model, instance_id, instance, result_path)
            rows.append(row)
            if stop_on_error and row["status"] == "error":
                raise RuntimeError(
                    f"Solver error for {row['model_name']} on {row['instance_id']}: "
                    f"{row['error']}"
                )
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    _run_single_model_instance,
                    model,
                    instance_id,
                    instance,
                    result_path,
                )
                for model, instance_id, instance, result_path in tasks
            ]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                if stop_on_error and row["status"] == "error":
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    raise RuntimeError(
                        f"Solver error for {row['model_name']} on {row['instance_id']}: "
                        f"{row['error']}"
                    )

    return pd.DataFrame(rows)


def load_results(outputs_dir=ROOT_DIR / "outputs"):
    """
    Load all result JSON files under outputs/ into a DataFrame.
    """
    out_root = Path(outputs_dir).resolve()
    if not out_root.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(out_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            payload.setdefault("model_name", path.parent.name)
            payload.setdefault("instance_id", path.stem)
            payload.setdefault("status", "ok")
            payload["result_file"] = str(path)
            rows.append(payload)
        except Exception:
            print(f"[load_results] skipping malformed JSON: {path}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for numeric_col in ("total_distance", "runtime_sec", "reported_runtime_sec"):
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
    return df


def compute_metrics(results_df):
    """
    Compute per-instance percent gap and model-level summary metrics.
    """
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = results_df.copy()
    if "status" not in df.columns:
        df["status"] = "ok"
    if "runtime_sec" not in df.columns and "reported_runtime_sec" in df.columns:
        df["runtime_sec"] = df["reported_runtime_sec"]
    if "total_distance" not in df.columns and "best_cost" in df.columns:
        df["total_distance"] = df["best_cost"]

    required_cols = {
        "model_name",
        "instance_id",
        "total_distance",
        "runtime_sec",
        "status",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"results_df is missing required columns: {sorted(missing)}")

    ok = df[df["status"] == "ok"].copy()
    ok = ok.dropna(subset=["total_distance", "runtime_sec"])
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    best_per_instance = (
        ok.groupby("instance_id", as_index=False)["total_distance"]
        .min()
        .rename(columns={"total_distance": "best_total_distance"})
    )

    detailed = ok.merge(best_per_instance, on="instance_id", how="left")
    detailed["percent_gap"] = (
        (detailed["total_distance"] - detailed["best_total_distance"])
        / detailed["best_total_distance"].replace(0, pd.NA)
        * 100.0
    )
    detailed["percent_gap"] = detailed["percent_gap"].fillna(0.0)

    summary = detailed.groupby("model_name", as_index=False).agg(
        instances=("instance_id", "nunique"),
        mean_total_distance=("total_distance", "mean"),
        median_total_distance=("total_distance", "median"),
        mean_runtime_sec=("runtime_sec", "mean"),
        median_runtime_sec=("runtime_sec", "median"),
        mean_percent_gap=("percent_gap", "mean"),
        median_percent_gap=("percent_gap", "median"),
        best_hits=("percent_gap", lambda s: int((s.abs() <= 1e-9).sum())),
    )

    attempted = df.groupby("model_name").size().rename("attempted_runs")
    succeeded = (
        df[df["status"] == "ok"].groupby("model_name").size().rename("successful_runs")
    )
    summary = summary.merge(attempted, on="model_name", how="left")
    summary = summary.merge(succeeded, on="model_name", how="left")
    summary["successful_runs"] = summary["successful_runs"].fillna(0).astype(int)
    summary["attempted_runs"] = summary["attempted_runs"].fillna(0).astype(int)
    summary["success_rate"] = (
        summary["successful_runs"] / summary["attempted_runs"].replace(0, pd.NA)
    ).fillna(0.0)
    summary = summary.sort_values(
        by=["mean_percent_gap", "mean_runtime_sec", "mean_total_distance"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return detailed, summary


def plot_results(
    detailed_df,
    summary_df,
    plots_dir=ROOT_DIR / "outputs" / "plots",
):
    """
    Create comparison plots and save them to disk.
    """
    if detailed_df.empty or summary_df.empty:
        return []

    plot_root = Path(plots_dir).resolve()
    plot_root.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    order = summary_df["model_name"].tolist()
    saved = []

    plt.figure(figsize=(max(10, len(order) * 1.1), 6))
    sns.boxplot(data=detailed_df, x="model_name", y="percent_gap", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percent Gap vs Best (%)")
    plt.xlabel("Model")
    plt.title("Solution Quality Comparison")
    plt.tight_layout()
    gap_path = plot_root / "percent_gap_boxplot.png"
    plt.savefig(gap_path, dpi=180)
    plt.close()
    saved.append(gap_path)

    plt.figure(figsize=(max(10, len(order) * 1.1), 6))
    sns.boxplot(data=detailed_df, x="model_name", y="runtime_sec", order=order)
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Runtime (sec, log scale)")
    plt.xlabel("Model")
    plt.title("Runtime Comparison")
    plt.tight_layout()
    rt_path = plot_root / "runtime_boxplot.png"
    plt.savefig(rt_path, dpi=180)
    plt.close()
    saved.append(rt_path)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=summary_df,
        x="mean_runtime_sec",
        y="mean_percent_gap",
        hue="model_name",
        s=100,
    )
    for _, row in summary_df.iterrows():
        plt.text(
            row["mean_runtime_sec"],
            row["mean_percent_gap"],
            row["model_name"],
            fontsize=8,
            ha="left",
            va="bottom",
        )
    plt.xscale("log")
    plt.xlabel("Mean Runtime (sec, log scale)")
    plt.ylabel("Mean Percent Gap (%)")
    plt.title("Quality vs Runtime Trade-off")
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    tradeoff_path = plot_root / "quality_runtime_tradeoff.png"
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()
    saved.append(tradeoff_path)

    return saved


def _parse_int_csv(value, field_name):
    items = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {field_name}: '{part}'") from exc
    if not items:
        raise ValueError(f"{field_name} must contain at least one integer")
    return items


def _parse_float_csv(value, field_name):
    items = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(float(part))
        except ValueError as exc:
            raise ValueError(f"Invalid float in {field_name}: '{part}'") from exc
    if not items:
        raise ValueError(f"{field_name} must contain at least one float")
    return items


def _dedupe_preserve(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _cycle_pick(values, index):
    return values[index % len(values)]


def _build_generation_knobs(args):
    knobs = {
        "fixed_customers": int(args.generated_fixed_customers),
        "fixed_cluster_size": int(args.generated_fixed_cluster_size),
        "max_coord": int(args.generated_max_coord),
        "customer_sizes": _dedupe_preserve(
            _parse_int_csv(args.generated_customer_sizes, "generated_customer_sizes")
        ),
        "cluster_seeds": _dedupe_preserve(
            _parse_int_csv(args.generated_cluster_seeds, "generated_cluster_seeds")
        ),
        "demand_types": _dedupe_preserve(
            _parse_int_csv(args.generated_demand_types, "generated_demand_types")
        ),
        "avg_route_sizes": _dedupe_preserve(
            _parse_int_csv(args.generated_avg_route_sizes, "generated_avg_route_sizes")
        ),
        "depot_positions": _dedupe_preserve(
            _parse_int_csv(args.generated_depot_positions, "generated_depot_positions")
        ),
        "customer_positions": _dedupe_preserve(
            _parse_int_csv(
                args.generated_customer_positions, "generated_customer_positions"
            )
        ),
        "decays": _dedupe_preserve(
            _parse_float_csv(args.generated_decays, "generated_decays")
        ),
        "random_fractions": _dedupe_preserve(
            _parse_float_csv(
                args.generated_random_fractions, "generated_random_fractions"
            )
        ),
    }

    if knobs["fixed_customers"] < 10:
        raise ValueError("generated_fixed_customers must be >= 10")
    if knobs["fixed_cluster_size"] < 1:
        raise ValueError("generated_fixed_cluster_size must be >= 1")
    if knobs["max_coord"] < 10:
        raise ValueError("generated_max_coord must be >= 10")

    for v in knobs["customer_sizes"]:
        if v < 10:
            raise ValueError("generated_customer_sizes values must be >= 10")
    for v in knobs["cluster_seeds"]:
        if v < 1:
            raise ValueError("generated_cluster_seeds values must be >= 1")
    for v in knobs["demand_types"]:
        if v < 1 or v > 7:
            raise ValueError("generated_demand_types values must be in [1, 7]")
    for v in knobs["avg_route_sizes"]:
        if v < 1 or v > 7:
            raise ValueError("generated_avg_route_sizes values must be in [1, 7]")
    for v in knobs["depot_positions"]:
        if v not in (1, 2, 3):
            raise ValueError("generated_depot_positions values must be 1, 2, or 3")
    for v in knobs["customer_positions"]:
        if v not in (1, 2, 3):
            raise ValueError("generated_customer_positions values must be 1, 2, or 3")
    for v in knobs["decays"]:
        if v <= 0:
            raise ValueError("generated_decays values must be > 0")
    for v in knobs["random_fractions"]:
        if v < 0.0 or v > 1.0:
            raise ValueError("generated_random_fractions values must be in [0.0, 1.0]")

    return knobs


def _profile_fixed_customers_variable_clusters(count, knobs):
    cases = []
    n_customers = knobs["fixed_customers"]

    for i in range(count):
        customer_pos = 2 if (i % 2 == 0) else 3
        demand_type = _cycle_pick(knobs["demand_types"], i)
        avg_route_size = _cycle_pick(knobs["avg_route_sizes"], i * 2)
        depot_pos = _cycle_pick(knobs["depot_positions"], i * 3)
        n_seeds = _cycle_pick(knobs["cluster_seeds"], i)
        decay = _cycle_pick(knobs["decays"], i * 5)
        random_fraction = 0.0
        if customer_pos == 3:
            random_fraction = _cycle_pick(knobs["random_fractions"], i * 7)

        cases.append(
            {
                "profile": "fixed-customers-variable-clusters",
                "n": n_customers,
                "depot_pos": depot_pos,
                "customer_pos": customer_pos,
                "demand_type": demand_type,
                "avg_route_size": avg_route_size,
                "n_seeds": n_seeds,
                "decay": decay,
                "random_fraction": random_fraction,
            }
        )

    return cases


def _profile_fixed_cluster_size_variable_customers(count, knobs):
    cases = []
    cluster_size = knobs["fixed_cluster_size"]

    for i in range(count):
        n_customers = _cycle_pick(knobs["customer_sizes"], i)
        customer_pos = 2 if (i % 3 != 0) else 3
        random_fraction = 0.0
        if customer_pos == 3:
            random_fraction = _cycle_pick(knobs["random_fractions"], i)

        clustered_fraction = (
            1.0 if customer_pos == 2 else max(0.05, 1.0 - random_fraction)
        )
        clustered_customers = max(1, int(round(n_customers * clustered_fraction)))

        estimated_seed_count = max(
            1, int(round(clustered_customers / float(cluster_size)))
        )
        estimated_seed_count += (i % 3) - 1
        estimated_seed_count = max(1, min(clustered_customers, estimated_seed_count))

        cases.append(
            {
                "profile": "fixed-cluster-size-variable-customers",
                "n": n_customers,
                "depot_pos": _cycle_pick(knobs["depot_positions"], i * 2),
                "customer_pos": customer_pos,
                "demand_type": _cycle_pick(knobs["demand_types"], i * 3),
                "avg_route_size": _cycle_pick(knobs["avg_route_sizes"], i * 5),
                "n_seeds": estimated_seed_count,
                "decay": _cycle_pick(knobs["decays"], i * 7),
                "random_fraction": random_fraction,
            }
        )

    return cases


def _profile_edge_cases(count, knobs):
    sizes = sorted(set(knobs["customer_sizes"] + [knobs["fixed_customers"]]))
    n_min = sizes[0]
    n_mid = sizes[len(sizes) // 2]
    n_max = sizes[-1]
    n_large = max(n_max, int(round(n_max * 1.5)))

    seed_min = min(knobs["cluster_seeds"])
    seed_max = max(knobs["cluster_seeds"])
    decay_min = min(knobs["decays"])
    decay_max = max(knobs["decays"])
    route_min = min(knobs["avg_route_sizes"])
    route_max = max(knobs["avg_route_sizes"])

    template = [
        {
            "n": n_min,
            "depot_pos": 3,
            "customer_pos": 2,
            "demand_type": 7,
            "avg_route_size": route_min,
            "n_seeds": seed_max,
            "decay": decay_min,
            "random_fraction": 0.0,
        },
        {
            "n": n_max,
            "depot_pos": 1,
            "customer_pos": 1,
            "demand_type": 4,
            "avg_route_size": route_max,
            "n_seeds": 1,
            "decay": decay_max,
            "random_fraction": 0.0,
        },
        {
            "n": n_mid,
            "depot_pos": 2,
            "customer_pos": 3,
            "demand_type": 6,
            "avg_route_size": route_min,
            "n_seeds": seed_max,
            "decay": decay_min,
            "random_fraction": min(knobs["random_fractions"]),
        },
        {
            "n": n_large,
            "depot_pos": 2,
            "customer_pos": 2,
            "demand_type": 5,
            "avg_route_size": route_max,
            "n_seeds": seed_min,
            "decay": decay_max,
            "random_fraction": 0.0,
        },
        {
            "n": n_max,
            "depot_pos": 3,
            "customer_pos": 3,
            "demand_type": 2,
            "avg_route_size": route_min,
            "n_seeds": seed_min,
            "decay": decay_max,
            "random_fraction": max(knobs["random_fractions"]),
        },
        {
            "n": n_mid,
            "depot_pos": 1,
            "customer_pos": 2,
            "demand_type": 1,
            "avg_route_size": route_max,
            "n_seeds": seed_max,
            "decay": decay_min,
            "random_fraction": 0.0,
        },
    ]

    cases = []
    for i in range(count):
        base = dict(template[i % len(template)])

        if i >= len(template):
            growth_factor = 1.0 + 0.1 * ((i // len(template)) % 4)
            base["n"] = max(10, int(round(base["n"] * growth_factor)))
            if base["customer_pos"] == 3:
                jitter = 0.05 * ((i % 5) - 2)
                base["random_fraction"] = max(
                    0.05, min(0.95, base["random_fraction"] + jitter)
                )

        base["profile"] = "edge-cases"
        cases.append(base)

    return cases


def _profile_mixed_stress(count, knobs, rng):
    cases = []

    n_candidates = list(knobs["customer_sizes"]) + [knobs["fixed_customers"]]
    n_candidates = sorted(set(n_candidates))

    for _ in range(count):
        n_customers = rng.choice(n_candidates)
        customer_pos = rng.choice(knobs["customer_positions"])
        demand_type = rng.choice(knobs["demand_types"])
        avg_route_size = rng.choice(knobs["avg_route_sizes"])
        depot_pos = rng.choice(knobs["depot_positions"])
        decay = rng.choice(knobs["decays"])

        if customer_pos == 3:
            random_fraction = rng.choice(knobs["random_fractions"])
        else:
            random_fraction = 0.0

        if customer_pos == 1:
            n_seeds = 1
        else:
            if rng.random() < 0.6:
                clustered_fraction = (
                    1.0 if customer_pos == 2 else max(0.05, 1.0 - random_fraction)
                )
                clustered_customers = max(
                    1, int(round(n_customers * clustered_fraction))
                )
                n_seeds = max(
                    1,
                    int(
                        round(clustered_customers / float(knobs["fixed_cluster_size"]))
                    ),
                )
            else:
                n_seeds = rng.choice(knobs["cluster_seeds"])

        cases.append(
            {
                "profile": "mixed-stress",
                "n": n_customers,
                "depot_pos": depot_pos,
                "customer_pos": customer_pos,
                "demand_type": demand_type,
                "avg_route_size": avg_route_size,
                "n_seeds": max(1, int(n_seeds)),
                "decay": float(decay),
                "random_fraction": random_fraction,
            }
        )

    return cases


def _build_generation_plan(args):
    profiles = args.generated_profile or ["mixed-stress"]
    profiles = _dedupe_preserve(profiles)

    total_instances = max(1, int(args.generated_num_instances))
    per_profile = total_instances // len(profiles)
    remainder = total_instances % len(profiles)

    knobs = _build_generation_knobs(args)
    rng = random.Random(int(args.generated_seed_base) + 991)

    cases = []
    for profile_idx, profile_name in enumerate(profiles):
        profile_count = per_profile + (1 if profile_idx < remainder else 0)
        if profile_count <= 0:
            continue

        if profile_name == "fixed-customers-variable-clusters":
            profile_cases = _profile_fixed_customers_variable_clusters(
                profile_count, knobs
            )
        elif profile_name == "fixed-cluster-size-variable-customers":
            profile_cases = _profile_fixed_cluster_size_variable_customers(
                profile_count, knobs
            )
        elif profile_name == "edge-cases":
            profile_cases = _profile_edge_cases(profile_count, knobs)
        elif profile_name == "mixed-stress":
            profile_cases = _profile_mixed_stress(profile_count, knobs, rng)
        else:
            raise ValueError(f"Unknown generation profile: {profile_name}")

        cases.extend(profile_cases)

    rng.shuffle(cases)

    seed_base = int(args.generated_seed_base)
    for idx, case in enumerate(cases):
        case["seed"] = seed_base + idx
        case["max_coord"] = knobs["max_coord"]

    return cases


def generate_benchmark_instances(args):
    """
    Generate synthetic benchmark instances using benchmarking/generatorLarge.py.
    Returns a summary dictionary with generated set metadata.
    """
    if generate_synthetic_instance is None:
        raise RuntimeError(
            "Could not import generator from benchmarking/generatorLarge.py. "
            "Fix that file first or disable --generate-instances."
        )

    cases = _build_generation_plan(args)
    if not cases:
        raise RuntimeError("Generation plan produced zero cases")

    sets_root = Path(args.sets_dir).resolve()
    set_name = args.generated_set_name
    set_dir = sets_root / set_name
    instances_dir = set_dir / "instances"
    solutions_dir = set_dir / "solutions"

    instances_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    if args.generated_clean:
        for vrp_file in instances_dir.glob("*.vrp"):
            vrp_file.unlink()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": str(
            (ROOT_DIR / "benchmarking" / "generatorLarge.py").resolve()
        ),
        "set_name": set_name,
        "set_dir": str(set_dir),
        "total_requested": len(cases),
        "profiles": args.generated_profile or ["mixed-stress"],
        "parameters": {
            "generated_num_instances": args.generated_num_instances,
            "generated_seed_base": args.generated_seed_base,
            "generated_fixed_customers": args.generated_fixed_customers,
            "generated_fixed_cluster_size": args.generated_fixed_cluster_size,
            "generated_customer_sizes": args.generated_customer_sizes,
            "generated_cluster_seeds": args.generated_cluster_seeds,
            "generated_demand_types": args.generated_demand_types,
            "generated_avg_route_sizes": args.generated_avg_route_sizes,
            "generated_depot_positions": args.generated_depot_positions,
            "generated_customer_positions": args.generated_customer_positions,
            "generated_decays": args.generated_decays,
            "generated_random_fractions": args.generated_random_fractions,
            "generated_max_coord": args.generated_max_coord,
            "generated_skip_tsplib95_validation": args.generated_skip_tsplib95_validation,
        },
        "successful": [],
        "failed": [],
    }

    for idx, case in enumerate(cases, start=1):
        output_stem = _slugify(
            f"gen_{idx:04d}_{case['profile']}_n{case['n']}_cp{case['customer_pos']}_"
            f"dp{case['depot_pos']}_dt{case['demand_type']}_rs{case['avg_route_size']}_"
            f"ns{case['n_seeds']}_s{case['seed']}"
        )

        try:
            metadata = generate_synthetic_instance(
                n_customers=int(case["n"]),
                depot_pos=int(case["depot_pos"]),
                customer_pos=int(case["customer_pos"]),
                demand_type=int(case["demand_type"]),
                avg_route_size=int(case["avg_route_size"]),
                rand_seed=int(case["seed"]),
                output_dir=instances_dir,
                output_name=output_stem,
                max_coord=int(case["max_coord"]),
                decay=float(case["decay"]),
                n_seeds=int(case["n_seeds"]),
                random_customer_fraction=float(case["random_fraction"]),
                overwrite=True,
                validate_tsplib95=not args.generated_skip_tsplib95_validation,
            )
            metadata["profile"] = case["profile"]
            metadata["plan"] = case
            manifest["successful"].append(metadata)
        except Exception as exc:
            failure = {
                "plan": case,
                "error": f"{type(exc).__name__}: {exc}",
            }
            manifest["failed"].append(failure)
            if args.generated_stop_on_error:
                manifest_path = set_dir / "generation_manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                raise

    manifest_path = set_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    success_count = len(manifest["successful"])
    fail_count = len(manifest["failed"])

    if success_count == 0:
        raise RuntimeError(
            "Instance generation failed for all planned cases. "
            f"See manifest: {manifest_path}"
        )

    return {
        "set_name": set_name,
        "instances_dir": str(instances_dir),
        "manifest_path": str(manifest_path),
        "success_count": success_count,
        "fail_count": fail_count,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run and analyze CVRP model benchmarks."
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT_DIR / "models")
    parser.add_argument("--sets-dir", type=Path, default=ROOT_DIR / "sets")
    parser.add_argument(
        "--set-name",
        action="append",
        dest="set_names",
        help="Set name to include. Repeat for multiple sets.",
    )
    parser.add_argument(
        "--n-amount",
        type=int,
        default=None,
        help="Number of instances to load (default: all from selected sets).",
    )
    parser.add_argument("--random-selection", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument(
        "--plots-dir", type=Path, default=OUTPUTS_ROOT / "plots"
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--detailed-csv", type=Path, default=None)

    parser.add_argument(
        "--model-pattern",
        type=str,
        default=None,
        help="Regex filter applied to discovered model names.",
    )
    parser.add_argument("--max-models", type=int, default=None)

    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only analyze existing outputs/ results.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing JSON result files."
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", help="Fail fast on first solver error."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List discovered models and exit."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker threads for solver execution.",
    )
    parser.add_argument(
        "--quiet-loader", action="store_true", help="Disable instance loader logs."
    )

    parser.add_argument(
        "--generate-instances",
        action="store_true",
        help="Generate synthetic instances with benchmarking/generatorLarge.py before benchmarking.",
    )
    parser.add_argument(
        "--include-existing-sets",
        action="store_true",
        help="With --generate-instances, also include any --set-name sets.",
    )
    parser.add_argument("--generated-set-name", type=str, default="GEN_XL")
    parser.add_argument(
        "--generated-clean",
        action="store_true",
        help="Delete old .vrp files in generated set.",
    )
    parser.add_argument("--generated-stop-on-error", action="store_true")
    parser.add_argument("--generated-num-instances", type=int, default=48)
    parser.add_argument("--generated-seed-base", type=int, default=10000)
    parser.add_argument(
        "--generated-profile",
        action="append",
        choices=[
            "fixed-customers-variable-clusters",
            "fixed-cluster-size-variable-customers",
            "edge-cases",
            "mixed-stress",
        ],
        help=(
            "Generation profile. Repeat to combine profiles. "
            "If omitted, defaults to mixed-stress."
        ),
    )

    parser.add_argument("--generated-fixed-customers", type=int, default=400)
    parser.add_argument("--generated-fixed-cluster-size", type=int, default=25)
    parser.add_argument(
        "--generated-customer-sizes", type=str, default="120,200,300,450,600"
    )
    parser.add_argument("--generated-cluster-seeds", type=str, default="2,4,6,8,10,12")
    parser.add_argument("--generated-demand-types", type=str, default="1,2,3,4,5,6,7")
    parser.add_argument(
        "--generated-avg-route-sizes", type=str, default="1,2,3,4,5,6,7"
    )
    parser.add_argument("--generated-depot-positions", type=str, default="1,2,3")
    parser.add_argument("--generated-customer-positions", type=str, default="1,2,3")
    parser.add_argument("--generated-decays", type=str, default="20,40,80")
    parser.add_argument("--generated-random-fractions", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--generated-max-coord", type=int, default=1000)
    parser.add_argument(
        "--generated-skip-tsplib95-validation",
        action="store_true",
        help="Skip tsplib95 validation for generated instances.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    outputs_dir, outputs_rebased = _normalize_output_path(args.outputs_dir)
    plots_dir, plots_rebased = _normalize_output_path(args.plots_dir)

    summary_candidate = (
        args.summary_csv if args.summary_csv is not None else outputs_dir / "benchmark_summary.csv"
    )
    detailed_candidate = (
        args.detailed_csv if args.detailed_csv is not None else outputs_dir / "benchmark_detailed.csv"
    )
    summary_csv, summary_rebased = _normalize_output_path(summary_candidate)
    detailed_csv, detailed_rebased = _normalize_output_path(detailed_candidate)

    if (
        outputs_rebased
        or plots_rebased
        or summary_rebased
        or detailed_rebased
    ):
        print("Output paths are constrained under outputs/:")
        if outputs_rebased:
            print(f"- results dir -> {outputs_dir}")
        if plots_rebased:
            print(f"- plots dir -> {plots_dir}")
        if summary_rebased:
            print(f"- summary csv -> {summary_csv}")
        if detailed_rebased:
            print(f"- detailed csv -> {detailed_csv}")

    models = discover_models(args.models_dir)

    if args.model_pattern:
        pattern = re.compile(args.model_pattern)
        models = [m for m in models if pattern.search(m.model_name)]

    if args.max_models is not None:
        models = models[: max(0, args.max_models)]

    if args.list_models:
        for model in models:
            print(
                f"{model.model_name} -> {model.source_file.name}:{model.function_name}"
            )
        print(f"Total discovered models: {len(models)}")
        return

    if not models:
        raise RuntimeError("No models were discovered. Nothing to run.")

    print(f"Discovered {len(models)} model(s)")

    if not args.skip_run:
        selected_set_names = list(args.set_names) if args.set_names else []

        if args.generate_instances:
            generation_summary = generate_benchmark_instances(args)
            print(
                "Generated synthetic instances using benchmarking/generatorLarge.py: "
                f"{generation_summary['success_count']} success, "
                f"{generation_summary['fail_count']} failed"
            )
            print(f"Generation manifest: {generation_summary['manifest_path']}")

            if args.include_existing_sets:
                selected_set_names = _dedupe_preserve(
                    selected_set_names + [generation_summary["set_name"]]
                )
            else:
                selected_set_names = [generation_summary["set_name"]]

        from benchmarking.load_cvrp import load_cvrp_instances

        instances = load_cvrp_instances(
            sets_dir=args.sets_dir,
            set_names=selected_set_names if selected_set_names else None,
            n_amount=args.n_amount,
            random_selection=args.random_selection,
            seed=args.seed,
            verbose=not args.quiet_loader,
        )
        if not instances:
            raise RuntimeError("No CVRP instances were loaded.")

        print(
            f"Running {len(models)} model(s) on {len(instances)} instance(s) "
            f"using {max(1, int(args.jobs))} thread(s)"
        )
        run_models(
            models=models,
            instances=instances,
            outputs_dir=outputs_dir,
            overwrite=args.overwrite,
            stop_on_error=args.stop_on_error,
            jobs=args.jobs,
        )
    else:
        if args.generate_instances:
            print("--generate-instances is ignored when --skip-run is enabled")

    results_df = load_results(outputs_dir)
    detailed_df, summary_df = compute_metrics(results_df)

    if detailed_df.empty or summary_df.empty:
        print(f"No successful benchmark results found in {outputs_dir}.")
        return

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detailed_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_csv, index=False)
    detailed_df.to_csv(detailed_csv, index=False)

    plot_paths = plot_results(detailed_df, summary_df, plots_dir)

    display_cols = [
        "model_name",
        "instances",
        "mean_total_distance",
        "mean_runtime_sec",
        "mean_percent_gap",
        "success_rate",
    ]
    present_cols = [c for c in display_cols if c in summary_df.columns]

    print("\nBenchmark summary:")
    print(summary_df[present_cols].to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved detailed CSV: {detailed_csv}")
    if plot_paths:
        print("Saved plots:")
        for p in plot_paths:
            print(f"- {p}")


if __name__ == "__main__":
    main()
