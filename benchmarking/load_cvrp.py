import argparse
import random
from pathlib import Path

import tsplib95


def _default_sets_dir():
    return Path(__file__).resolve().parents[1] / "sets"


def _resolve_set_dirs(sets_path, set_names):
    if set_names is None:
        return sorted(path for path in sets_path.iterdir() if path.is_dir())

    selected = []
    for set_name in set_names:
        set_path = sets_path / set_name
        if not set_path.exists():
            raise FileNotFoundError(f"Set '{set_name}' was not found at {set_path}")
        if not set_path.is_dir():
            raise NotADirectoryError(f"Set '{set_name}' is not a directory: {set_path}")
        selected.append(set_path)
    return selected


def _discover_pairs(sets_path, set_names):
    pairs = []
    for set_dir in _resolve_set_dirs(sets_path, set_names):
        instances_dir = set_dir / "instances"
        solutions_dir = set_dir / "solutions"

        if not instances_dir.exists():
            continue

        for instance_path in sorted(instances_dir.glob("*.vrp")):
            solution_path = solutions_dir / f"{instance_path.stem}.sol"
            pairs.append(
                (
                    set_dir.name,
                    instance_path,
                    solution_path if solution_path.exists() else None,
                )
            )
    return pairs


def _select_pairs(
    pairs,
    n_amount,
    random_selection,
    random_n_amount,
    seed,
):
    if not pairs:
        return []

    rng = random.Random(seed)

    if random_n_amount:
        upper_bound = len(pairs) if n_amount is None else min(n_amount, len(pairs))
        if upper_bound < 1:
            raise ValueError("n_amount must be >= 1 when random_n_amount is enabled")
        n_amount = rng.randint(1, upper_bound)

    if n_amount is None:
        n_amount = len(pairs)

    if n_amount < 1:
        raise ValueError("n_amount must be >= 1")
    if n_amount > len(pairs):
        raise ValueError(
            f"Requested n_amount={n_amount}, but only {len(pairs)} instances are available"
        )

    if random_selection or random_n_amount:
        return rng.sample(pairs, n_amount)
    return pairs[:n_amount]


def load_cvrp_dataset(
    sets_dir=None,
    set_names=None,
    n_amount=None,
    random_selection=False,
    random_n_amount=False,
    seed=None,
    verbose=True,
):
    """
    Load CVRP instances from a sets directory organized as:
        sets/<SET_NAME>/instances/*.vrp
        sets/<SET_NAME>/solutions/*.sol

    Returns a dictionary keyed by '<set_name>/<instance_stem>', with:
        - problem: tsplib95 loaded problem object
        - set_name: set folder name (e.g., 'A', 'XL')
        - instance_path: Path to the .vrp file
        - solution_path: Path to matching .sol file, or None if missing
    """
    sets_path = Path(sets_dir) if sets_dir is not None else _default_sets_dir()
    sets_path = sets_path.resolve()

    if not sets_path.exists():
        raise FileNotFoundError(f"Directory {sets_path} not found")

    pairs = _discover_pairs(sets_path, set_names)
    selected_pairs = _select_pairs(
        pairs=pairs,
        n_amount=n_amount,
        random_selection=random_selection,
        random_n_amount=random_n_amount,
        seed=seed,
    )

    dataset = {}
    for set_name, instance_path, solution_path in selected_pairs:
        try:
            problem = tsplib95.load(str(instance_path))
        except Exception as exc:
            if verbose:
                print(f"Error loading {instance_path}: {exc}")
            continue

        key = f"{set_name}/{instance_path.stem}"
        dataset[key] = {
            "problem": problem,
            "set_name": set_name,
            "instance_path": instance_path,
            "solution_path": solution_path,
        }
        if verbose:
            has_solution = "yes" if solution_path is not None else "no"
            print(f"Loaded: {key} (solution: {has_solution})")

    return dataset


def load_cvrp_instances(
    sets_dir=None,
    set_names=None,
    n_amount=None,
    random_selection=False,
    random_n_amount=False,
    seed=None,
    verbose=True,
):
    """Backward-compatible loader that returns only tsplib95 problem objects."""
    dataset = load_cvrp_dataset(
        sets_dir=sets_dir,
        set_names=set_names,
        n_amount=n_amount,
        random_selection=random_selection,
        random_n_amount=random_n_amount,
        seed=seed,
        verbose=verbose,
    )
    return {name: item["problem"] for name, item in dataset.items()}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Load CVRP instances from nested set folders."
    )
    parser.add_argument(
        "--sets-dir",
        type=Path,
        default=_default_sets_dir(),
        help="Path to the root sets directory (default: project_root/sets)",
    )
    parser.add_argument(
        "--set-name",
        action="append",
        dest="set_names",
        help="Set name to include (repeat for multiple sets, e.g. --set-name A --set-name X)",
    )
    parser.add_argument(
        "--n-amount",
        type=int,
        default=None,
        help="How many instances to load. If omitted, load all selected instances.",
    )
    parser.add_argument(
        "--random-selection",
        action="store_true",
        help="Pick the selected instances randomly instead of taking the first n_amount.",
    )
    parser.add_argument(
        "--random-n-amount",
        action="store_true",
        help="Randomly choose n_amount in [1, n_amount] (or [1, total] if n_amount omitted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random selection / random n_amount.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-instance logging.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    dataset = load_cvrp_dataset(
        sets_dir=args.sets_dir,
        set_names=args.set_names,
        n_amount=args.n_amount,
        random_selection=args.random_selection,
        random_n_amount=args.random_n_amount,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(f"\nTotal instances loaded: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    main()
