"""
Memetic Algorithm (MA) for CVRP.

Hybrid approach:
- Global exploration with a Genetic Algorithm.
- Local intensification with intra/inter improvement heuristics.
"""

import importlib.util
import random
from pathlib import Path

try:
    from models.constructive import (
        cheapest_insertion,
        clarke_wright,
        nearest_neighbor,
        sweep,
    )
    from models.utils import copy_routes, extract_instance_data, solution_cost
except ImportError:
    from constructive import cheapest_insertion, clarke_wright, nearest_neighbor, sweep
    from utils import copy_routes, extract_instance_data, solution_cost


def _load_module_from_file(filename, module_name):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INTRA_MOD = _load_module_from_file("improve-intra.py", "improve_intra_mod_for_memetic")
_INTER_MOD = _load_module_from_file("improve-inter.py", "improve_inter_mod_for_memetic")


def _build_maps(instance):
    _, _, demands, capacity, customer_ids = extract_instance_data(instance)
    demand = {customer_ids[i]: demands[i] for i in range(len(customer_ids))}
    customer_set = set(customer_ids)
    return demand, capacity, customer_ids, customer_set


def _routes_to_permutation(routes):
    return [node for route in routes for node in route]


def _decode_permutation(perm, demand, capacity):
    routes = []
    route = []
    load = 0
    for node in perm:
        d = demand[node]
        if load + d <= capacity:
            route.append(node)
            load += d
        else:
            if route:
                routes.append(route)
            route = [node]
            load = d
    if route:
        routes.append(route)
    return routes


def _valid_permutation(perm, customer_set):
    return len(perm) == len(customer_set) and set(perm) == customer_set


def _evaluate(instance, perm, demand, capacity):
    routes = _decode_permutation(perm, demand, capacity)
    return solution_cost(instance, routes), routes


def _tournament_select(population, fitness, k, rng):
    idxs = rng.sample(range(len(population)), min(k, len(population)))
    best = min(idxs, key=lambda i: fitness[i])
    return list(population[best])


def _order_crossover(parent_a, parent_b, rng):
    n = len(parent_a)
    if n < 2:
        return list(parent_a)
    i, j = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[i : j + 1] = parent_a[i : j + 1]
    taken = set(child[i : j + 1])
    fill = [g for g in parent_b if g not in taken]
    ptr = 0
    for pos in range(n):
        if child[pos] is None:
            child[pos] = fill[ptr]
            ptr += 1
    return child


def _mutate_swap(perm, rng):
    if len(perm) < 2:
        return perm
    i, j = rng.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def _mutate_invert(perm, rng):
    if len(perm) < 2:
        return perm
    i, j = sorted(rng.sample(range(len(perm)), 2))
    perm[i : j + 1] = reversed(perm[i : j + 1])
    return perm


def _mutate_relocate(perm, rng):
    if len(perm) < 2:
        return perm
    i, j = rng.sample(range(len(perm)), 2)
    node = perm.pop(i)
    perm.insert(j, node)
    return perm


def _mutate(perm, rng):
    op = rng.choice(("swap", "invert", "relocate"))
    if op == "swap":
        return _mutate_swap(perm, rng)
    if op == "invert":
        return _mutate_invert(perm, rng)
    return _mutate_relocate(perm, rng)


def _intensify(instance, routes, lam, intra_methods, inter_methods, intra_passes, inter_passes):
    current = copy_routes(routes)
    current = _INTRA_MOD.improve_intra(
        instance,
        current,
        methods=intra_methods,
        max_passes=intra_passes,
    )
    current = _INTER_MOD.improve_inter(
        instance,
        current,
        methods=inter_methods,
        max_passes=inter_passes,
        lam=lam,
    )
    return current


def _seed_population(instance, population_size, customer_ids, customer_set, rng):
    heuristics = [nearest_neighbor, clarke_wright, sweep, cheapest_insertion]
    population = []
    seen = set()

    def _try_add(perm):
        key = tuple(perm)
        if key in seen:
            return
        if not _valid_permutation(perm, customer_set):
            return
        population.append(list(perm))
        seen.add(key)

    for fn in heuristics:
        try:
            perm = _routes_to_permutation(fn(instance))
            _try_add(perm)
        except Exception:
            pass

    genes = list(customer_ids)
    while len(population) < population_size:
        perm = list(genes)
        rng.shuffle(perm)
        _try_add(perm)
        if len(population) < population_size and len(seen) > 4 * population_size:
            population.append(list(perm))

    return population[:population_size]


def _educate_individual(
    instance,
    perm,
    demand,
    capacity,
    lam,
    intra_methods,
    inter_methods,
    intra_passes,
    inter_passes,
):
    routes = _decode_permutation(perm, demand, capacity)
    improved = _intensify(
        instance,
        routes,
        lam,
        intra_methods,
        inter_methods,
        intra_passes,
        inter_passes,
    )
    return _routes_to_permutation(improved)


def memetic_algorithm(
    instance,
    initial_routes=None,
    population_size=50,
    generations=200,
    crossover_rate=0.9,
    mutation_rate=0.20,
    tournament_size=3,
    elitism=2,
    seed=42,
    lam=3,
    intra_methods=("2opt", "or_opt", "relocate", "exchange"),
    inter_methods=("insert", "swap", "cross"),
    offspring_local_search_prob=0.35,
    elite_local_search_count=2,
    elite_local_search_period=5,
    intra_passes=1,
    inter_passes=1,
    diversification=True,
    stagnation_limit=30,
    restart_fraction=0.25,
    return_history=True,
):
    """
    Memetic Algorithm for CVRP minimization.
    """
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if elitism < 0 or elitism >= population_size:
        raise ValueError("elitism must be in [0, population_size - 1]")

    rng = random.Random(seed)
    demand, capacity, customer_ids, customer_set = _build_maps(instance)

    if any(demand[cid] > capacity for cid in customer_ids):
        raise ValueError("At least one customer demand exceeds vehicle capacity.")

    population = _seed_population(instance, population_size, customer_ids, customer_set, rng)

    if initial_routes is not None:
        init_perm = _routes_to_permutation(initial_routes)
        if _valid_permutation(init_perm, customer_set):
            population[0] = init_perm

    fitness = []
    decoded = []
    for perm in population:
        c, r = _evaluate(instance, perm, demand, capacity)
        fitness.append(c)
        decoded.append(r)

    best_idx = min(range(len(population)), key=lambda i: fitness[i])
    best_perm = list(population[best_idx])
    best_cost = fitness[best_idx]
    history = []
    no_improve = 0

    for gen in range(1, generations + 1):
        ranking = sorted(range(len(population)), key=lambda i: fitness[i])
        new_population = [list(population[i]) for i in ranking[:elitism]]

        while len(new_population) < population_size:
            p1 = _tournament_select(population, fitness, tournament_size, rng)
            p2 = _tournament_select(population, fitness, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = _order_crossover(p1, p2, rng)
            else:
                child = list(p1)

            if rng.random() < mutation_rate:
                child = _mutate(child, rng)

            if rng.random() < offspring_local_search_prob:
                child = _educate_individual(
                    instance,
                    child,
                    demand,
                    capacity,
                    lam,
                    intra_methods,
                    inter_methods,
                    intra_passes,
                    inter_passes,
                )

            if not _valid_permutation(child, customer_set):
                child = list(customer_ids)
                rng.shuffle(child)

            new_population.append(child)

        population = new_population

        # Periodic elite education
        if elite_local_search_count > 0 and elite_local_search_period > 0 and gen % elite_local_search_period == 0:
            ranking = sorted(range(len(population)), key=lambda i: _evaluate(instance, population[i], demand, capacity)[0])
            to_educate = ranking[: min(elite_local_search_count, len(population))]
            for idx in to_educate:
                population[idx] = _educate_individual(
                    instance,
                    population[idx],
                    demand,
                    capacity,
                    lam,
                    intra_methods,
                    inter_methods,
                    intra_passes,
                    inter_passes,
                )

        fitness = []
        decoded = []
        for perm in population:
            c, r = _evaluate(instance, perm, demand, capacity)
            fitness.append(c)
            decoded.append(r)

        gen_best_idx = min(range(len(population)), key=lambda i: fitness[i])
        gen_best_cost = fitness[gen_best_idx]

        if gen_best_cost < best_cost - 1e-9:
            best_cost = gen_best_cost
            best_perm = list(population[gen_best_idx])
            no_improve = 0
        else:
            no_improve += 1

        if diversification and no_improve >= stagnation_limit:
            ranking = sorted(range(len(population)), key=lambda i: fitness[i])
            restart_count = max(1, int(restart_fraction * population_size))
            replace_idxs = ranking[-restart_count:]
            for idx in replace_idxs:
                perm = list(customer_ids)
                rng.shuffle(perm)
                population[idx] = perm
            no_improve = 0

        if return_history:
            avg_cost = sum(fitness) / len(fitness)
            unique_ratio = len({tuple(ind) for ind in population}) / float(len(population))
            history.append(
                {
                    "generation": gen,
                    "best_cost": best_cost,
                    "generation_best_cost": gen_best_cost,
                    "avg_cost": avg_cost,
                    "unique_ratio": unique_ratio,
                }
            )

    best_routes = _decode_permutation(best_perm, demand, capacity)
    return {
        "best_routes": best_routes,
        "best_cost": best_cost,
        "best_chromosome": best_perm,
        "generations": generations,
        "history": history if return_history else None,
    }


def memetic_from_nearest_neighbor(instance, **kwargs):
    """
    Convenience wrapper that seeds MA with nearest-neighbor routes.
    """
    init_routes = nearest_neighbor(instance)
    return memetic_algorithm(instance, initial_routes=init_routes, **kwargs)


__all__ = ["memetic_algorithm", "memetic_from_nearest_neighbor"]
