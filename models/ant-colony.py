"""
Ant Colony Optimization (ACO) for CVRP.

Construction:
- Each ant builds a feasible CVRP solution by probabilistically selecting
  the next customer using pheromone and heuristic information.
- If no feasible customer fits the remaining capacity, the ant starts a new route.
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


_INTRA_MOD = _load_module_from_file("improve-intra.py", "improve_intra_mod_for_aco")
_INTER_MOD = _load_module_from_file("improve-inter.py", "improve_inter_mod_for_aco")


def _build_maps(instance):
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    if any(d > capacity for d in demands):
        raise ValueError("At least one customer demand exceeds vehicle capacity.")

    coords = {customer_ids[i]: customers[i] for i in range(len(customer_ids))}
    demand = {customer_ids[i]: demands[i] for i in range(len(customer_ids))}
    customer_set = set(customer_ids)
    return depot, coords, demand, capacity, customer_ids, customer_set


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


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_distance_matrix(depot, coords, customer_ids):
    """
    Internal index map:
    - 0 => depot
    - i+1 => customer_ids[i]
    """
    points = [depot] + [coords[cid] for cid in customer_ids]
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(points[i], points[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _build_inverse_distance(dist, epsilon=1e-9):
    n = len(dist)
    eta = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                eta[i][j] = 0.0
            else:
                eta[i][j] = 1.0 / max(dist[i][j], epsilon)
    return eta


def _select_next_customer(feasible_nodes, scores, rng):
    total = sum(scores)
    if total <= 0:
        return rng.choice(feasible_nodes)
    r = rng.random() * total
    acc = 0.0
    for node, score in zip(feasible_nodes, scores):
        acc += score
        if acc >= r:
            return node
    return feasible_nodes[-1]


def _construct_ant_solution(
    customer_ids,
    demand,
    capacity,
    tau,
    eta,
    alpha,
    beta,
    q0,
    rng,
):
    """
    Build one feasible solution:
    - route as list of customer ids
    """
    unvisited = set(customer_ids)
    routes = []

    id_to_internal = {cid: idx + 1 for idx, cid in enumerate(customer_ids)}

    while unvisited:
        route = []
        rem_cap = capacity
        current_internal = 0  # depot

        while True:
            feasible = [cid for cid in unvisited if demand[cid] <= rem_cap]
            if not feasible:
                break

            feasible_internal = [id_to_internal[cid] for cid in feasible]
            desirability = []
            for nxt in feasible_internal:
                val = (tau[current_internal][nxt] ** alpha) * (eta[current_internal][nxt] ** beta)
                desirability.append(val)

            if rng.random() < q0:
                # Exploitation
                idx = max(range(len(feasible)), key=lambda i: desirability[i])
                chosen = feasible[idx]
            else:
                # Exploration (roulette)
                chosen = _select_next_customer(feasible, desirability, rng)

            route.append(chosen)
            unvisited.remove(chosen)
            rem_cap -= demand[chosen]
            current_internal = id_to_internal[chosen]

        if not route:
            # Should not happen due demand <= capacity checks.
            raise RuntimeError("Ant failed to construct a feasible non-empty route.")
        routes.append(route)

    return _cleanup(routes)


def _deposit_pheromone_on_solution(solution, delta_tau, tau, customer_ids):
    id_to_internal = {cid: idx + 1 for idx, cid in enumerate(customer_ids)}
    for route in solution:
        prev = 0  # depot
        for cid in route:
            curr = id_to_internal[cid]
            tau[prev][curr] += delta_tau
            tau[curr][prev] += delta_tau
            prev = curr
        tau[prev][0] += delta_tau
        tau[0][prev] += delta_tau


def _bound_pheromone(tau, tau_min, tau_max):
    n = len(tau)
    for i in range(n):
        for j in range(n):
            if tau[i][j] < tau_min:
                tau[i][j] = tau_min
            elif tau[i][j] > tau_max:
                tau[i][j] = tau_max


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


def ant_colony_optimization(
    instance,
    initial_routes=None,
    num_ants=35,
    iterations=250,
    alpha=1.0,
    beta=3.0,
    evaporation_rate=0.15,
    q0=0.20,
    pheromone_init=1.0,
    elite_weight=2.0,
    seed=42,
    use_local_search=False,
    local_search_prob=0.15,
    lam=3,
    pheromone_bounds=(1e-4, 1e3),
    return_history=True,
):
    """
    Ant Colony Optimization for CVRP minimization.
    """
    if num_ants < 1:
        raise ValueError("num_ants must be >= 1")
    if not (0.0 <= evaporation_rate < 1.0):
        raise ValueError("evaporation_rate must be in [0, 1)")
    if not (0.0 <= q0 <= 1.0):
        raise ValueError("q0 must be in [0, 1]")
    if pheromone_init <= 0:
        raise ValueError("pheromone_init must be > 0")

    rng = random.Random(seed)
    depot, coords, demand, capacity, customer_ids, customer_set = _build_maps(instance)

    dist = _build_distance_matrix(depot, coords, customer_ids)
    eta = _build_inverse_distance(dist)
    n_internal = len(customer_ids) + 1
    tau = [[pheromone_init for _ in range(n_internal)] for _ in range(n_internal)]

    if initial_routes is None:
        current_seed = nearest_neighbor(instance)
    else:
        current_seed = copy_routes(initial_routes)

    if not _feasible_solution(current_seed, demand, capacity, customer_set):
        raise ValueError("Initial routes are not feasible for this CVRP instance.")

    best_routes = copy_routes(current_seed)
    best_cost = solution_cost(instance, best_routes)
    history = []

    tau_min, tau_max = pheromone_bounds

    for it in range(1, iterations + 1):
        ant_solutions = []
        ant_costs = []

        for _ in range(num_ants):
            ant_routes = _construct_ant_solution(
                customer_ids,
                demand,
                capacity,
                tau,
                eta,
                alpha,
                beta,
                q0,
                rng,
            )

            if use_local_search and rng.random() < local_search_prob:
                ant_routes = _intensify(instance, ant_routes, True, lam)

            if not _feasible_solution(ant_routes, demand, capacity, customer_set):
                continue

            c = solution_cost(instance, ant_routes)
            ant_solutions.append(ant_routes)
            ant_costs.append(c)

        if not ant_solutions:
            # Recovery: keep pheromone alive and continue.
            for i in range(n_internal):
                for j in range(n_internal):
                    tau[i][j] *= (1.0 - evaporation_rate)
            _bound_pheromone(tau, tau_min, tau_max)
            continue

        iter_best_idx = min(range(len(ant_solutions)), key=lambda i: ant_costs[i])
        iter_best_routes = ant_solutions[iter_best_idx]
        iter_best_cost = ant_costs[iter_best_idx]

        if iter_best_cost < best_cost - 1e-9:
            best_cost = iter_best_cost
            best_routes = copy_routes(iter_best_routes)

        # Evaporation
        evap = 1.0 - evaporation_rate
        for i in range(n_internal):
            for j in range(n_internal):
                tau[i][j] *= evap

        # Deposit from iteration best and global best.
        delta_iter = 1.0 / max(iter_best_cost, 1e-12)
        _deposit_pheromone_on_solution(iter_best_routes, delta_iter, tau, customer_ids)

        delta_global = elite_weight / max(best_cost, 1e-12)
        _deposit_pheromone_on_solution(best_routes, delta_global, tau, customer_ids)

        _bound_pheromone(tau, tau_min, tau_max)

        if return_history:
            history.append(
                {
                    "iteration": it,
                    "iteration_best_cost": iter_best_cost,
                    "global_best_cost": best_cost,
                }
            )

    return {
        "best_routes": best_routes,
        "best_cost": best_cost,
        "iterations": iterations,
        "history": history if return_history else None,
    }


def aco_from_nearest_neighbor(instance, **kwargs):
    """
    Convenience wrapper that initializes from nearest-neighbor solution.
    """
    init_routes = nearest_neighbor(instance)
    return ant_colony_optimization(instance, initial_routes=init_routes, **kwargs)


__all__ = ["ant_colony_optimization", "aco_from_nearest_neighbor"]
