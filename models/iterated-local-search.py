"""
Iterated Local Search (ILS) for CVRP.

Flow:
1) Build an initial feasible solution.
2) Improve it with local search.
3) Repeatedly perturb + local search.
4) Accept/reject candidate and keep global best.
"""

import importlib.util
import math
import random
from pathlib import Path

try:
    from models.constructive import nearest_neighbor
    from models.utils import copy_routes, extract_instance_data, solution_cost
except ImportError:
    from constructive import nearest_neighbor
    from utils import copy_routes, extract_instance_data, solution_cost


def _load_module_from_file(filename, module_name):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INTRA_MOD = _load_module_from_file("improve-intra.py", "improve_intra_mod")
_INTER_MOD = _load_module_from_file("improve-inter.py", "improve_inter_mod")


def _build_maps(instance):
    _, _, demands, capacity, customer_ids = extract_instance_data(instance)
    demand = {customer_ids[i]: demands[i] for i in range(len(customer_ids))}
    customer_set = set(customer_ids)
    return demand, capacity, customer_set


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


def _pick_route_index(routes, rng, min_size=1, different_from=None):
    valid = [idx for idx, route in enumerate(routes) if len(route) >= min_size]
    if different_from is not None:
        valid = [idx for idx in valid if idx != different_from]
    if not valid:
        return None
    return rng.choice(valid)


def _move_intra_2opt(routes, rng):
    new_routes = copy_routes(routes)
    ridx = _pick_route_index(new_routes, rng, min_size=4)
    if ridx is None:
        return None
    route = new_routes[ridx]
    i = rng.randint(0, len(route) - 3)
    j = rng.randint(i + 2, len(route) - 1)
    route[i : j + 1] = reversed(route[i : j + 1])
    return _cleanup(new_routes)


def _move_intra_relocate(routes, rng):
    new_routes = copy_routes(routes)
    ridx = _pick_route_index(new_routes, rng, min_size=2)
    if ridx is None:
        return None
    route = new_routes[ridx]
    i = rng.randrange(len(route))
    node = route.pop(i)
    j = rng.randrange(len(route) + 1)
    route.insert(j, node)
    return _cleanup(new_routes)


def _move_inter_insert(routes, rng):
    new_routes = copy_routes(routes)
    src = _pick_route_index(new_routes, rng, min_size=1)
    if src is None:
        return None
    dst = _pick_route_index(new_routes, rng, min_size=0, different_from=src)
    if dst is None:
        return None

    src_route = new_routes[src]
    dst_route = new_routes[dst]
    i = rng.randrange(len(src_route))
    node = src_route.pop(i)
    j = rng.randrange(len(dst_route) + 1)
    dst_route.insert(j, node)
    return _cleanup(new_routes)


def _move_inter_swap(routes, rng):
    new_routes = copy_routes(routes)
    a = _pick_route_index(new_routes, rng, min_size=1)
    if a is None:
        return None
    b = _pick_route_index(new_routes, rng, min_size=1, different_from=a)
    if b is None:
        return None
    ra = new_routes[a]
    rb = new_routes[b]
    ia = rng.randrange(len(ra))
    ib = rng.randrange(len(rb))
    ra[ia], rb[ib] = rb[ib], ra[ia]
    return _cleanup(new_routes)


def _move_two_opt_star(routes, rng):
    new_routes = copy_routes(routes)
    a = _pick_route_index(new_routes, rng, min_size=1)
    if a is None:
        return None
    b = _pick_route_index(new_routes, rng, min_size=1, different_from=a)
    if b is None:
        return None
    ra = new_routes[a]
    rb = new_routes[b]

    cut_a = rng.randrange(len(ra) + 1)
    cut_b = rng.randrange(len(rb) + 1)
    new_routes[a] = ra[:cut_a] + rb[cut_b:]
    new_routes[b] = rb[:cut_b] + ra[cut_a:]
    return _cleanup(new_routes)


def _neighbor_factory():
    return {
        "intra_2opt": lambda routes, rng: _move_intra_2opt(routes, rng),
        "intra_relocate": lambda routes, rng: _move_intra_relocate(routes, rng),
        "inter_insert": lambda routes, rng: _move_inter_insert(routes, rng),
        "inter_swap": lambda routes, rng: _move_inter_swap(routes, rng),
        "two_opt_star": lambda routes, rng: _move_two_opt_star(routes, rng),
    }


def _perturb(
    routes,
    move_map,
    perturbation_strength,
    rng,
    demand,
    capacity,
    customer_set,
    moves,
):
    perturbed = copy_routes(routes)
    for _ in range(max(1, perturbation_strength)):
        move_name = rng.choice(moves)
        candidate = move_map[move_name](perturbed, rng)
        if candidate is None:
            continue
        if _feasible_solution(candidate, demand, capacity, customer_set):
            perturbed = candidate
    return perturbed


def _local_search(
    instance,
    routes,
    local_search_rounds,
    intra_methods,
    inter_methods,
    lam,
):
    current = copy_routes(routes)
    for _ in range(max(1, local_search_rounds)):
        before = solution_cost(instance, current)
        current = _INTRA_MOD.improve_intra(
            instance,
            current,
            methods=intra_methods,
            max_passes=1,
        )
        current = _INTER_MOD.improve_inter(
            instance,
            current,
            methods=inter_methods,
            max_passes=1,
            lam=lam,
        )
        after = solution_cost(instance, current)
        if after >= before - 1e-9:
            break
    return current


def iterated_local_search(
    instance,
    initial_routes=None,
    max_iterations=100,
    perturbation_strength=3,
    local_search_rounds=1,
    seed=42,
    acceptance="better",
    acceptance_temperature=1.0,
    cooling_rate=0.999,
    lam=3,
    intra_methods=("2opt", "or_opt", "relocate", "exchange"),
    inter_methods=("insert", "swap", "cross"),
    perturbation_moves=("intra_2opt", "intra_relocate", "inter_insert", "inter_swap", "two_opt_star"),
    diversification=True,
    stagnation_limit=80,
    max_perturbation_strength=10,
    return_history=True,
):
    """
    Iterated Local Search for CVRP.

    acceptance:
    - "better": accept candidate only if it improves current
    - "metropolis": SA-like probabilistic acceptance for worse candidates
    """
    rng = random.Random(seed)
    demand, capacity, customer_set = _build_maps(instance)
    move_map = _neighbor_factory()

    unknown = [name for name in perturbation_moves if name not in move_map]
    if unknown:
        raise ValueError("Unknown perturbation move(s): " + ", ".join(unknown))

    if initial_routes is None:
        current = nearest_neighbor(instance)
    else:
        current = copy_routes(initial_routes)

    if not _feasible_solution(current, demand, capacity, customer_set):
        raise ValueError("Initial routes are not feasible for this CVRP instance.")

    current = _local_search(
        instance, current, local_search_rounds, intra_methods, inter_methods, lam
    )
    current_cost = solution_cost(instance, current)
    best = copy_routes(current)
    best_cost = current_cost

    temp = max(1e-9, float(acceptance_temperature))
    history = []
    no_improve = 0
    strength = max(1, int(perturbation_strength))

    for it in range(1, max_iterations + 1):
        perturbed = _perturb(
            current,
            move_map,
            strength,
            rng,
            demand,
            capacity,
            customer_set,
            perturbation_moves,
        )

        candidate = _local_search(
            instance,
            perturbed,
            local_search_rounds,
            intra_methods,
            inter_methods,
            lam,
        )

        if not _feasible_solution(candidate, demand, capacity, customer_set):
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

        if cand_cost < best_cost - 1e-9:
            best = copy_routes(candidate)
            best_cost = cand_cost
            no_improve = 0
            strength = max(1, perturbation_strength)
        else:
            no_improve += 1

        if diversification and no_improve >= stagnation_limit:
            strength = min(max_perturbation_strength, strength + 1)
            no_improve = 0

        temp *= cooling_rate

        if return_history:
            history.append(
                {
                    "iteration": it,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "perturbation_strength": strength,
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


def ils_from_nearest_neighbor(instance, **kwargs):
    """
    Convenience wrapper: starts from nearest-neighbor solution.
    """
    init_routes = nearest_neighbor(instance)
    return iterated_local_search(instance, initial_routes=init_routes, **kwargs)


__all__ = ["iterated_local_search", "ils_from_nearest_neighbor"]
