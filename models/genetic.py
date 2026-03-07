"""
Genetic Algorithm (GA) for CVRP.

Representation:
- Chromosome is a permutation of all customers (giant tour without depots).
- Decoder splits the permutation into feasible routes under vehicle capacity.
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


_INTRA_MOD = _load_module_from_file("improve-intra.py", "improve_intra_mod_for_ga")
_INTER_MOD = _load_module_from_file("improve-inter.py", "improve_inter_mod_for_ga")


def _build_maps(instance):
    _, _, demands, capacity, customer_ids = extract_instance_data(instance)
    demand = {customer_ids[i]: demands[i] for i in range(len(customer_ids))}
    customer_set = set(customer_ids)
    return demand, capacity, customer_ids, customer_set


def _routes_to_permutation(routes):
    return [node for route in routes for node in route]


def _decode_permutation(permutation, demand, capacity):
    routes = []
    route = []
    load = 0

    for node in permutation:
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
    cost = solution_cost(instance, routes)
    return cost, routes


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

    fill_values = [gene for gene in parent_b if gene not in taken]
    fill_idx = 0
    for pos in range(n):
        if child[pos] is None:
            child[pos] = fill_values[fill_idx]
            fill_idx += 1
    return child


def _mutate_swap(perm, rng):
    n = len(perm)
    if n < 2:
        return perm
    i, j = rng.sample(range(n), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def _mutate_invert(perm, rng):
    n = len(perm)
    if n < 2:
        return perm
    i, j = sorted(rng.sample(range(n), 2))
    perm[i : j + 1] = reversed(perm[i : j + 1])
    return perm


def _mutate_relocate(perm, rng):
    n = len(perm)
    if n < 2:
        return perm
    i, j = rng.sample(range(n), 2)
    node = perm.pop(i)
    perm.insert(j, node)
    return perm


def _mutate(perm, rng):
    choice = rng.choice(("swap", "invert", "relocate"))
    if choice == "swap":
        return _mutate_swap(perm, rng)
    if choice == "invert":
        return _mutate_invert(perm, rng)
    return _mutate_relocate(perm, rng)


def _intensify(instance, routes, use_local_search, lam, intra_methods, inter_methods):
    if not use_local_search:
        return routes
    current = copy_routes(routes)
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
    return current


def _seed_population(instance, population_size, customer_ids, customer_set, rng, demand, capacity):
    heuristics = [nearest_neighbor, clarke_wright, sweep, cheapest_insertion]
    genes = list(customer_ids)
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
            routes = fn(instance)
            perm = _routes_to_permutation(routes)
            _try_add(perm)
        except Exception:
            pass

    while len(population) < population_size:
        perm = list(genes)
        rng.shuffle(perm)
        _try_add(perm)
        if len(population) < population_size and len(seen) > 3 * population_size:
            # If diversity is saturated, allow duplicates to finish quickly.
            population.append(list(perm))

    return population[:population_size]


def genetic_algorithm(
    instance,
    initial_routes=None,
    population_size=50,
    generations=250,
    crossover_rate=0.9,
    mutation_rate=0.25,
    tournament_size=3,
    elitism=2,
    seed=42,
    use_local_search=False,
    local_search_prob=0.15,
    lam=3,
    intra_methods=("2opt", "or_opt", "relocate", "exchange"),
    inter_methods=("insert", "swap", "cross"),
    diversification=True,
    stagnation_limit=35,
    restart_fraction=0.30,
    return_history=True,
):
    """
    Genetic Algorithm for CVRP minimization.
    """
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if elitism < 0 or elitism >= population_size:
        raise ValueError("elitism must be in [0, population_size - 1]")

    rng = random.Random(seed)
    demand, capacity, customer_ids, customer_set = _build_maps(instance)

    if any(demand[cid] > capacity for cid in customer_ids):
        raise ValueError("At least one customer demand exceeds vehicle capacity.")

    population = _seed_population(
        instance, population_size, customer_ids, customer_set, rng, demand, capacity
    )

    if initial_routes is not None:
        init_perm = _routes_to_permutation(initial_routes)
        if _valid_permutation(init_perm, customer_set):
            population[0] = init_perm

    fitness = []
    decoded = []
    for perm in population:
        cost, routes = _evaluate(instance, perm, demand, capacity)
        fitness.append(cost)
        decoded.append(routes)

    best_idx = min(range(len(population)), key=lambda i: fitness[i])
    best_perm = list(population[best_idx])
    best_routes = copy_routes(decoded[best_idx])
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

            if use_local_search and rng.random() < local_search_prob:
                child_routes = _decode_permutation(child, demand, capacity)
                child_routes = _intensify(
                    instance, child_routes, True, lam, intra_methods, inter_methods
                )
                child = _routes_to_permutation(child_routes)

            if not _valid_permutation(child, customer_set):
                child = list(customer_ids)
                rng.shuffle(child)

            new_population.append(child)

        population = new_population
        fitness = []
        decoded = []
        for perm in population:
            cost, routes = _evaluate(instance, perm, demand, capacity)
            fitness.append(cost)
            decoded.append(routes)

        gen_best_idx = min(range(len(population)), key=lambda i: fitness[i])
        gen_best_cost = fitness[gen_best_idx]

        if gen_best_cost < best_cost - 1e-9:
            best_cost = gen_best_cost
            best_perm = list(population[gen_best_idx])
            best_routes = copy_routes(decoded[gen_best_idx])
            no_improve = 0
        else:
            no_improve += 1

        if diversification and no_improve >= stagnation_limit:
            # Restart a fraction of worst individuals.
            ranking = sorted(range(len(population)), key=lambda i: fitness[i])
            restart_count = max(1, int(restart_fraction * population_size))
            replace_idxs = ranking[-restart_count:]
            for idx in replace_idxs:
                perm = list(customer_ids)
                rng.shuffle(perm)
                population[idx] = perm
                cost, routes = _evaluate(instance, perm, demand, capacity)
                fitness[idx] = cost
                decoded[idx] = routes
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

    # Decode from best chromosome for consistency.
    final_best_routes = _decode_permutation(best_perm, demand, capacity)
    return {
        "best_routes": final_best_routes,
        "best_cost": best_cost,
        "best_chromosome": best_perm,
        "generations": generations,
        "history": history if return_history else None,
    }


def ga_from_nearest_neighbor(instance, **kwargs):
    """
    Convenience wrapper that seeds GA with nearest-neighbor routes.
    """
    init_routes = nearest_neighbor(instance)
    return genetic_algorithm(instance, initial_routes=init_routes, **kwargs)


__all__ = ["genetic_algorithm", "ga_from_nearest_neighbor"]
