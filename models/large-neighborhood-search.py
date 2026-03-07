"""
Large Neighborhood Search (LNS) for CVRP.

This implementation uses:
- Destroy operators: random removal, related removal, worst removal
- Repair operators: greedy insertion, regret-2 insertion
- Optional intensification with intra/inter local search modules
"""

import importlib.util
import math
import random
from pathlib import Path

try:
    from models.constructive import nearest_neighbor
    from models.utils import copy_routes, extract_instance_data, route_cost, solution_cost
except ImportError:
    from constructive import nearest_neighbor
    from utils import copy_routes, extract_instance_data, route_cost, solution_cost


def _load_module_from_file(filename, module_name):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INTRA_MOD = _load_module_from_file("improve-intra.py", "improve_intra_mod_for_lns")
_INTER_MOD = _load_module_from_file("improve-inter.py", "improve_inter_mod_for_lns")


def _build_maps(instance):
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    coords = {customer_ids[i]: customers[i] for i in range(len(customer_ids))}
    demand = {customer_ids[i]: demands[i] for i in range(len(customer_ids))}
    customer_set = set(customer_ids)
    return depot, coords, demand, capacity, customer_set


def _route_load(route, demand):
    return sum(demand[node] for node in route)


def _cleanup(routes):
    return [route for route in routes if route]


def _feasible_solution(routes, demand, capacity, customer_set):
    flat = [node for route in routes for node in route]
    if len(flat) != len(customer_set):
        return False
    if set(flat) != customer_set:
        return False
    if any(_route_load(route, demand) > capacity for route in routes):
        return False
    return True


def _flatten_customers(routes):
    return [node for route in routes for node in route]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _remove_nodes(routes, node_set):
    out = []
    for route in routes:
        new_route = [node for node in route if node not in node_set]
        if new_route:
            out.append(new_route)
    return out


def _destroy_random(routes, q, rng):
    all_nodes = _flatten_customers(routes)
    if not all_nodes:
        return copy_routes(routes), []
    q = min(q, len(all_nodes))
    removed = rng.sample(all_nodes, q)
    partial = _remove_nodes(routes, set(removed))
    return partial, removed


def _destroy_related(routes, q, rng, coords):
    all_nodes = _flatten_customers(routes)
    if not all_nodes:
        return copy_routes(routes), []
    q = min(q, len(all_nodes))
    seed = rng.choice(all_nodes)
    remaining = set(all_nodes)
    removed = [seed]
    remaining.remove(seed)

    while len(removed) < q and remaining:
        candidate = min(
            remaining,
            key=lambda node: min(_dist(coords[node], coords[r]) for r in removed),
        )
        removed.append(candidate)
        remaining.remove(candidate)

    partial = _remove_nodes(routes, set(removed))
    return partial, removed


def _removal_gain(route, pos, depot, coords):
    node = route[pos]
    prev_coord = depot if pos == 0 else coords[route[pos - 1]]
    curr_coord = coords[node]
    next_coord = depot if pos == len(route) - 1 else coords[route[pos + 1]]
    old = _dist(prev_coord, curr_coord) + _dist(curr_coord, next_coord)
    new = _dist(prev_coord, next_coord)
    return old - new


def _destroy_worst(routes, q, rng, depot, coords):
    candidates = []
    for ridx, route in enumerate(routes):
        for pos, node in enumerate(route):
            gain = _removal_gain(route, pos, depot, coords)
            # Small noise for diversification among similar gains.
            gain += rng.random() * 1e-4
            candidates.append((gain, node))

    if not candidates:
        return copy_routes(routes), []

    candidates.sort(reverse=True, key=lambda item: item[0])
    q = min(q, len(candidates))
    removed = [node for _, node in candidates[:q]]
    partial = _remove_nodes(routes, set(removed))
    return partial, removed


def _insertion_delta(route, pos, node, depot, coords):
    prev_coord = depot if pos == 0 else coords[route[pos - 1]]
    next_coord = depot if pos == len(route) else coords[route[pos]]
    node_coord = coords[node]
    return _dist(prev_coord, node_coord) + _dist(node_coord, next_coord) - _dist(prev_coord, next_coord)


def _best_insertions_for_node(node, routes, demand, capacity, depot, coords):
    insertions = []
    for ridx, route in enumerate(routes):
        if _route_load(route, demand) + demand[node] > capacity:
            continue
        for pos in range(len(route) + 1):
            delta = _insertion_delta(route, pos, node, depot, coords)
            insertions.append((delta, ridx, pos))

    # Option: new dedicated route.
    new_route_delta = 2.0 * _dist(depot, coords[node])
    insertions.append((new_route_delta, len(routes), 0))
    insertions.sort(key=lambda x: x[0])
    return insertions


def _apply_insertion(routes, node, ridx, pos):
    if ridx == len(routes):
        routes.append([node])
    else:
        routes[ridx].insert(pos, node)


def _repair_greedy(partial_routes, removed_nodes, demand, capacity, depot, coords):
    routes = copy_routes(partial_routes)
    pool = list(removed_nodes)
    while pool:
        best = None
        for node in pool:
            insertions = _best_insertions_for_node(node, routes, demand, capacity, depot, coords)
            if not insertions:
                continue
            delta, ridx, pos = insertions[0]
            if best is None or delta < best[0]:
                best = (delta, node, ridx, pos)

        if best is None:
            return None

        _, node, ridx, pos = best
        _apply_insertion(routes, node, ridx, pos)
        pool.remove(node)

    return _cleanup(routes)


def _repair_regret2(partial_routes, removed_nodes, demand, capacity, depot, coords):
    routes = copy_routes(partial_routes)
    pool = list(removed_nodes)

    while pool:
        chosen = None
        for node in pool:
            insertions = _best_insertions_for_node(node, routes, demand, capacity, depot, coords)
            if not insertions:
                continue
            best = insertions[0][0]
            second = insertions[1][0] if len(insertions) > 1 else best + 1e6
            regret = second - best
            if chosen is None or regret > chosen[0]:
                _, ridx, pos = insertions[0]
                chosen = (regret, node, ridx, pos)

        if chosen is None:
            return None

        _, node, ridx, pos = chosen
        _apply_insertion(routes, node, ridx, pos)
        pool.remove(node)

    return _cleanup(routes)


def _intensify(instance, routes, use_local_search, lam):
    if not use_local_search:
        return routes
    current = copy_routes(routes)
    current = _INTRA_MOD.improve_intra(
        instance,
        current,
        methods=("2opt", "or_opt", "relocate", "exchange"),
        max_passes=1,
    )
    current = _INTER_MOD.improve_inter(
        instance,
        current,
        methods=("insert", "swap", "cross"),
        max_passes=1,
        lam=lam,
    )
    return current


def large_neighborhood_search(
    instance,
    initial_routes=None,
    max_iterations=400,
    min_destroy_fraction=0.10,
    max_destroy_fraction=0.30,
    seed=42,
    acceptance="metropolis",
    acceptance_temperature=10.0,
    cooling_rate=0.998,
    lam=3,
    use_local_search=True,
    local_search_every=1,
    diversification=True,
    stagnation_limit=60,
    return_history=True,
):
    """
    Large Neighborhood Search for CVRP.

    acceptance:
    - "better": accept only improving candidate
    - "metropolis": probabilistic acceptance for worse candidates
    """
    rng = random.Random(seed)
    depot, coords, demand, capacity, customer_set = _build_maps(instance)

    if initial_routes is None:
        current = nearest_neighbor(instance)
    else:
        current = copy_routes(initial_routes)

    if not _feasible_solution(current, demand, capacity, customer_set):
        raise ValueError("Initial routes are not feasible for this CVRP instance.")

    current = _intensify(instance, current, use_local_search, lam)
    current_cost = solution_cost(instance, current)
    best = copy_routes(current)
    best_cost = current_cost

    destroy_ops = ("random", "related", "worst")
    repair_ops = ("greedy", "regret2")
    destroy_scores = [1.0, 1.0, 1.0]
    repair_scores = [1.0, 1.0]
    temp = max(1e-9, float(acceptance_temperature))
    history = []
    no_improve = 0

    for it in range(1, max_iterations + 1):
        n_customers = len(_flatten_customers(current))
        min_q = max(1, int(min_destroy_fraction * n_customers))
        max_q = max(min_q, int(max_destroy_fraction * n_customers))
        q = rng.randint(min_q, max_q)

        # Roulette wheel for adaptive operator selection.
        destroy_name = rng.choices(destroy_ops, weights=destroy_scores, k=1)[0]
        repair_name = rng.choices(repair_ops, weights=repair_scores, k=1)[0]

        if destroy_name == "random":
            partial, removed = _destroy_random(current, q, rng)
            d_idx = 0
        elif destroy_name == "related":
            partial, removed = _destroy_related(current, q, rng, coords)
            d_idx = 1
        else:
            partial, removed = _destroy_worst(current, q, rng, depot, coords)
            d_idx = 2

        if repair_name == "greedy":
            candidate = _repair_greedy(partial, removed, demand, capacity, depot, coords)
            r_idx = 0
        else:
            candidate = _repair_regret2(partial, removed, demand, capacity, depot, coords)
            r_idx = 1

        if candidate is None:
            destroy_scores[d_idx] *= 0.98
            repair_scores[r_idx] *= 0.98
            continue

        if (it % max(1, local_search_every)) == 0:
            candidate = _intensify(instance, candidate, use_local_search, lam)

        if not _feasible_solution(candidate, demand, capacity, customer_set):
            destroy_scores[d_idx] *= 0.97
            repair_scores[r_idx] *= 0.97
            continue

        cand_cost = solution_cost(instance, candidate)
        delta = cand_cost - current_cost
        accept = False

        if acceptance == "better":
            accept = delta < -1e-9
        elif acceptance == "metropolis":
            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / max(temp, 1e-12))
                if rng.random() < prob:
                    accept = True
        else:
            raise ValueError("acceptance must be 'better' or 'metropolis'")

        if accept:
            current = candidate
            current_cost = cand_cost
            destroy_scores[d_idx] *= 1.01
            repair_scores[r_idx] *= 1.01
        else:
            destroy_scores[d_idx] *= 0.999
            repair_scores[r_idx] *= 0.999

        if cand_cost < best_cost - 1e-9:
            best = copy_routes(candidate)
            best_cost = cand_cost
            no_improve = 0
            destroy_scores[d_idx] *= 1.03
            repair_scores[r_idx] *= 1.03
        else:
            no_improve += 1

        # Diversification: temporarily increase destroy strength by raising min fraction.
        if diversification and no_improve >= stagnation_limit:
            min_destroy_fraction = min(0.45, min_destroy_fraction + 0.03)
            max_destroy_fraction = min(0.70, max_destroy_fraction + 0.05)
            no_improve = 0
        else:
            # Gradually recover baseline neighborhood size.
            min_destroy_fraction = max(0.10, min_destroy_fraction * 0.999)
            max_destroy_fraction = max(0.30, max_destroy_fraction * 0.999)

        temp *= cooling_rate

        if return_history:
            history.append(
                {
                    "iteration": it,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "destroy_op": destroy_name,
                    "repair_op": repair_name,
                    "q": q,
                    "temperature": temp,
                }
            )

    return {
        "best_routes": best,
        "best_cost": best_cost,
        "final_routes": current,
        "final_cost": current_cost,
        "iterations": max_iterations,
        "history": history if return_history else None,
    }


def lns_from_nearest_neighbor(instance, **kwargs):
    """
    Convenience wrapper: starts from nearest-neighbor solution.
    """
    init_routes = nearest_neighbor(instance)
    return large_neighborhood_search(instance, initial_routes=init_routes, **kwargs)


__all__ = ["large_neighborhood_search", "lns_from_nearest_neighbor"]
