import argparse
import json
import math
import random
from pathlib import Path

try:
    import tsplib95
except Exception:
    tsplib95 = None


ROUTE_SIZE_INTERVALS = {
    1: (3, 5),
    2: (5, 8),
    3: (8, 12),
    4: (12, 16),
    5: (16, 25),
    6: (25, 50),
    7: (50, 200),
}

DEMAND_MIN_VALUES = [1, 1, 5, 1, 50, 1, 51, 50, 1]
DEMAND_MAX_VALUES = [1, 10, 10, 100, 100, 50, 100, 100, 10]


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _random_unique_point(rng, max_coord, used_points, forbidden_point=None):
    while True:
        point = (rng.randint(0, max_coord), rng.randint(0, max_coord))
        if point in used_points:
            continue
        if forbidden_point is not None and point == forbidden_point:
            continue
        return point


def _validate_inputs(
    n_customers,
    depot_pos,
    customer_pos,
    demand_type,
    avg_route_size,
    max_coord,
    decay,
    n_seeds,
    random_customer_fraction,
):
    if n_customers < 1:
        raise ValueError("n_customers must be >= 1")
    if depot_pos not in (1, 2, 3):
        raise ValueError("depot_pos must be one of: 1, 2, 3")
    if customer_pos not in (1, 2, 3):
        raise ValueError("customer_pos must be one of: 1, 2, 3")
    if demand_type < 1 or demand_type > 7:
        raise ValueError("demand_type must be in [1, 7]")
    if avg_route_size < 1 or avg_route_size > 7:
        raise ValueError("avg_route_size must be in [1, 7]")
    if max_coord < 10:
        raise ValueError("max_coord must be >= 10")
    if decay <= 0:
        raise ValueError("decay must be > 0")
    if n_seeds is not None and n_seeds < 1:
        raise ValueError("n_seeds must be >= 1 when provided")
    if not (0.0 <= random_customer_fraction <= 1.0):
        raise ValueError("random_customer_fraction must be in [0.0, 1.0]")


def _validate_tsplib95_instance(output_path, expected_dimension, expected_capacity):
    if tsplib95 is None:
        raise RuntimeError(
            "tsplib95 is not installed. Install it or pass validate_tsplib95=False."
        )

    problem = tsplib95.load(str(output_path))

    if int(problem.dimension) != int(expected_dimension):
        raise ValueError(
            f"Generated file failed dimension check: "
            f"{problem.dimension} != {expected_dimension}"
        )
    if int(problem.capacity) != int(expected_capacity):
        raise ValueError(
            f"Generated file failed capacity check: "
            f"{problem.capacity} != {expected_capacity}"
        )
    if str(problem.type).upper() != "CVRP":
        raise ValueError(f"Generated file type is not CVRP: {problem.type}")


def generate_instance(
    n_customers,
    depot_pos,
    customer_pos,
    demand_type,
    avg_route_size,
    rand_seed,
    output_dir=".",
    output_name=None,
    max_coord=1000,
    decay=40.0,
    n_seeds=None,
    random_customer_fraction=0.5,
    overwrite=False,
    validate_tsplib95=True,
):
    """
    Generate one CVRP .vrp instance inspired by Uchoa et al. style synthetic generation.

    Parameters follow the original generator convention:
    - depot_pos: 1=random, 2=centered, 3=cornered
    - customer_pos: 1=random, 2=clustered, 3=random-clustered
    - demand_type: 1..7
    - avg_route_size: 1..7
    """
    _validate_inputs(
        n_customers=n_customers,
        depot_pos=depot_pos,
        customer_pos=customer_pos,
        demand_type=demand_type,
        avg_route_size=avg_route_size,
        max_coord=max_coord,
        decay=decay,
        n_seeds=n_seeds,
        random_customer_fraction=random_customer_fraction,
    )

    rng = random.Random(rand_seed)

    route_low, route_high = ROUTE_SIZE_INTERVALS[avg_route_size]
    route_size_factor = rng.uniform(route_low, route_high)

    if depot_pos == 1:
        depot = (rng.randint(0, max_coord), rng.randint(0, max_coord))
    elif depot_pos == 2:
        depot = (int(max_coord / 2.0), int(max_coord / 2.0))
    else:
        depot = (0, 0)

    used_points = set()
    customers = []

    if customer_pos == 1:
        n_random_customers = n_customers
        n_clustered_customers = 0
        n_seeds_used = 0
    elif customer_pos == 2:
        n_random_customers = 0
        n_clustered_customers = n_customers
        n_seeds_used = n_seeds if n_seeds is not None else rng.randint(2, 6)
    else:
        n_random_customers = int(round(n_customers * random_customer_fraction))
        n_random_customers = min(max(n_random_customers, 0), n_customers)
        n_clustered_customers = n_customers - n_random_customers
        n_seeds_used = n_seeds if n_seeds is not None else rng.randint(2, 6)

    for _ in range(n_random_customers):
        point = _random_unique_point(rng, max_coord, used_points, forbidden_point=depot)
        used_points.add(point)
        customers.append(point)

    seeds = []
    if n_clustered_customers > 0:
        n_seeds_used = min(max(1, n_seeds_used), n_clustered_customers)

        for _ in range(n_seeds_used):
            point = _random_unique_point(
                rng, max_coord, used_points, forbidden_point=depot
            )
            used_points.add(point)
            seeds.append(point)
            customers.append(point)

        max_weight = 0.0
        for seed_point in seeds:
            weight_sum = 0.0
            for other_seed in seeds:
                weight_sum += 2 ** (-distance(seed_point, other_seed) / decay)
            max_weight = max(max_weight, weight_sum)
        norm_factor = 1.0 / max(max_weight, 1e-12)

        target_total_customers = n_customers
        accepted_count = len(customers)
        attempts = 0
        max_attempts = max(10000, n_customers * 500)

        while accepted_count < target_total_customers and attempts < max_attempts:
            attempts += 1
            point = _random_unique_point(
                rng, max_coord, used_points, forbidden_point=depot
            )

            weight = 0.0
            for seed_point in seeds:
                weight += 2 ** (-distance(point, seed_point) / decay)
            weight *= norm_factor

            if rng.random() <= min(1.0, weight):
                used_points.add(point)
                customers.append(point)
                accepted_count += 1

        while accepted_count < target_total_customers:
            point = _random_unique_point(
                rng, max_coord, used_points, forbidden_point=depot
            )
            used_points.add(point)
            customers.append(point)
            accepted_count += 1

    customers = sorted(customers)
    vertices = [depot] + customers

    demand_min = DEMAND_MIN_VALUES[demand_type - 1]
    demand_max = DEMAND_MAX_VALUES[demand_type - 1]

    demand_min_even_quadrant = 51
    demand_max_even_quadrant = 100
    demand_min_large = 50
    demand_max_large = 100
    demand_min_small = 1
    demand_max_small = 10
    large_per_route = 1.5

    demands = []
    sum_demands = 0
    max_demand = 0

    threshold_large = (n_customers / route_size_factor) * large_per_route

    for idx in range(2, n_customers + 2):
        customer_point = vertices[idx - 1]

        demand = rng.randint(demand_min, demand_max)

        if demand_type == 6:
            same_half = (
                customer_point[0] < max_coord / 2.0
                and customer_point[1] < max_coord / 2.0
            ) or (
                customer_point[0] >= max_coord / 2.0
                and customer_point[1] >= max_coord / 2.0
            )
            if same_half:
                demand = rng.randint(demand_min_even_quadrant, demand_max_even_quadrant)

        if demand_type == 7:
            if idx < threshold_large:
                demand = rng.randint(demand_min_large, demand_max_large)
            else:
                demand = rng.randint(demand_min_small, demand_max_small)

        demands.append(demand)
        max_demand = max(max_demand, demand)
        sum_demands += demand

    if sum_demands == n_customers:
        capacity = math.floor(route_size_factor)
    else:
        capacity = max(
            max_demand, math.ceil(route_size_factor * sum_demands / n_customers)
        )

    k = int(math.ceil(sum_demands / float(capacity)))

    if output_name is None:
        instance_name = f"XLTEST-n{n_customers + 1}-k{k}-s{rand_seed}"
    else:
        instance_name = output_name

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"{instance_name}.vrp"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    if demand_type != 6:
        rng.shuffle(demands)

    demands_with_depot = [0] + demands

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"NAME : {instance_name}\n")
        f.write(
            'COMMENT : "Synthetic CVRP instance generated with generatorLarge-compatible logic."\n'
        )
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {n_customers + 1}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {int(capacity)}\n")
        f.write("NODE_COORD_SECTION\n")

        for i, vertex in enumerate(vertices):
            f.write(f"{i + 1:<4} {vertex[0]:<4} {vertex[1]:<4}\n")

        f.write("DEMAND_SECTION\n")
        for i, _ in enumerate(vertices):
            f.write(f"{i + 1:<4} {demands_with_depot[i]:<4}\n")

        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

    if validate_tsplib95:
        try:
            _validate_tsplib95_instance(
                output_path=output_path,
                expected_dimension=n_customers + 1,
                expected_capacity=int(capacity),
            )
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

    metadata = {
        "instance_name": instance_name,
        "output_path": str(output_path),
        "n_customers": n_customers,
        "dimension": n_customers + 1,
        "estimated_vehicles": k,
        "capacity": int(capacity),
        "sum_demands": int(sum_demands),
        "generation": {
            "depot_pos": depot_pos,
            "customer_pos": customer_pos,
            "demand_type": demand_type,
            "avg_route_size": avg_route_size,
            "rand_seed": rand_seed,
            "max_coord": max_coord,
            "decay": decay,
            "n_seeds": n_seeds_used,
            "random_customer_fraction": random_customer_fraction,
            "n_random_customers": n_random_customers,
            "n_clustered_customers": n_clustered_customers,
        },
        "validated_with_tsplib95": bool(validate_tsplib95),
    }

    return metadata


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic CVRP instance (generatorLarge-compatible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("n", type=int, help="Number of customers")
    parser.add_argument(
        "depot_pos", type=int, help="Depot position: 1=random, 2=centered, 3=cornered"
    )
    parser.add_argument(
        "customer_pos",
        type=int,
        help="Customer position: 1=random, 2=clustered, 3=random-clustered",
    )
    parser.add_argument("demand_type", type=int, help="Demand type in [1..7]")
    parser.add_argument(
        "avg_route_size", type=int, help="Average route size class in [1..7]"
    )
    parser.add_argument("rand_seed", type=int, help="Random seed")

    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Directory for .vrp output"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output file stem without extension",
    )
    parser.add_argument(
        "--max-coord", type=int, default=1000, help="Maximum coordinate value"
    )
    parser.add_argument(
        "--decay", type=float, default=40.0, help="Cluster decay parameter"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=None, help="Override number of cluster seeds"
    )
    parser.add_argument(
        "--random-customer-fraction",
        type=float,
        default=0.5,
        help="Random customer fraction for customer_pos=3",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file"
    )
    parser.add_argument(
        "--skip-tsplib95-validation",
        action="store_true",
        help="Skip post-write validation with tsplib95.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print generation metadata as JSON"
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    metadata = generate_instance(
        n_customers=args.n,
        depot_pos=args.depot_pos,
        customer_pos=args.customer_pos,
        demand_type=args.demand_type,
        avg_route_size=args.avg_route_size,
        rand_seed=args.rand_seed,
        output_dir=args.output_dir,
        output_name=args.output_name,
        max_coord=args.max_coord,
        decay=args.decay,
        n_seeds=args.n_seeds,
        random_customer_fraction=args.random_customer_fraction,
        overwrite=args.overwrite,
        validate_tsplib95=not args.skip_tsplib95_validation,
    )

    if args.json:
        print(json.dumps(metadata, indent=2))
    else:
        print(
            f"Generated {metadata['instance_name']} -> {metadata['output_path']} "
            f"(n={metadata['n_customers']}, k~{metadata['estimated_vehicles']})"
        )


if __name__ == "__main__":
    main()
