"""
Inter-route improvement heuristics for CVRP.

Includes:
- 2-opt* (inter-route edge exchange)
- Insert (inter-route relocate)
- Swap (inter-route exchange)
- CROSS (exchange short strings between routes)
- lambda-interchange (exchange arbitrary subsets up to lambda)
"""

import itertools

try:
    from models.utils import (
        copy_routes,
        extract_instance_data,
        route_cost,
        solution_cost,
        validate_demands_capacity,
    )
except ImportError:
    from utils import (
        copy_routes,
        extract_instance_data,
        route_cost,
        solution_cost,
        validate_demands_capacity,
    )


def _build_maps(instance):
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    validate_demands_capacity(demands, capacity)
    coords = {customer_ids[i]: customers[i] for i in range(len(customer_ids))}
    demand = {customer_ids[i]: demands[i] for i in range(len(customer_ids))}
    return depot, coords, demand, capacity


def _route_load(route, demand):
    return sum(demand[node] for node in route)


def _cleanup_empty_routes(routes):
    return [route for route in routes if route]


def _iterate_solution_improvement(routes, move_fn, max_iterations):
    current = copy_routes(routes)
    for _ in range(max_iterations):
        delta, candidate = move_fn(current)
        if delta < -1e-9:
            current = candidate
        else:
            break
    return current


def _best_two_opt_star_move(routes, depot, coords, demand, capacity):
    best_delta = 0.0
    best_routes = routes

    for ra in range(len(routes) - 1):
        for rb in range(ra + 1, len(routes)):
            route_a = routes[ra]
            route_b = routes[rb]
            base_cost = route_cost(route_a, depot, coords) + route_cost(
                route_b, depot, coords
            )

            for cut_a in range(len(route_a) + 1):
                head_a = route_a[:cut_a]
                tail_a = route_a[cut_a:]
                for cut_b in range(len(route_b) + 1):
                    head_b = route_b[:cut_b]
                    tail_b = route_b[cut_b:]

                    cand_a = head_a + tail_b
                    cand_b = head_b + tail_a

                    if _route_load(cand_a, demand) > capacity:
                        continue
                    if _route_load(cand_b, demand) > capacity:
                        continue

                    cand_cost = route_cost(cand_a, depot, coords) + route_cost(
                        cand_b, depot, coords
                    )
                    delta = cand_cost - base_cost
                    if delta < best_delta:
                        candidate = copy_routes(routes)
                        candidate[ra] = cand_a
                        candidate[rb] = cand_b
                        best_delta = delta
                        best_routes = _cleanup_empty_routes(candidate)

    return best_delta, best_routes


def _best_insert_move(routes, depot, coords, demand, capacity):
    best_delta = 0.0
    best_routes = routes

    for ra in range(len(routes)):
        route_a = routes[ra]
        if not route_a:
            continue
        for rb in range(len(routes)):
            if ra == rb:
                continue
            route_b = routes[rb]
            base_cost = route_cost(route_a, depot, coords) + route_cost(
                route_b, depot, coords
            )

            for i in range(len(route_a)):
                node = route_a[i]
                cand_a = route_a[:i] + route_a[i + 1 :]
                for j in range(len(route_b) + 1):
                    cand_b = route_b[:j] + [node] + route_b[j:]

                    if _route_load(cand_a, demand) > capacity:
                        continue
                    if _route_load(cand_b, demand) > capacity:
                        continue

                    cand_cost = route_cost(cand_a, depot, coords) + route_cost(
                        cand_b, depot, coords
                    )
                    delta = cand_cost - base_cost
                    if delta < best_delta:
                        candidate = copy_routes(routes)
                        candidate[ra] = cand_a
                        candidate[rb] = cand_b
                        best_delta = delta
                        best_routes = _cleanup_empty_routes(candidate)

    return best_delta, best_routes


def _best_swap_move(routes, depot, coords, demand, capacity):
    best_delta = 0.0
    best_routes = routes

    for ra in range(len(routes) - 1):
        for rb in range(ra + 1, len(routes)):
            route_a = routes[ra]
            route_b = routes[rb]
            if not route_a or not route_b:
                continue

            base_cost = route_cost(route_a, depot, coords) + route_cost(
                route_b, depot, coords
            )
            base_load_a = _route_load(route_a, demand)
            base_load_b = _route_load(route_b, demand)

            for i in range(len(route_a)):
                for j in range(len(route_b)):
                    node_a = route_a[i]
                    node_b = route_b[j]
                    cand_load_a = base_load_a - demand[node_a] + demand[node_b]
                    cand_load_b = base_load_b - demand[node_b] + demand[node_a]
                    if cand_load_a > capacity or cand_load_b > capacity:
                        continue

                    cand_a = list(route_a)
                    cand_b = list(route_b)
                    cand_a[i], cand_b[j] = cand_b[j], cand_a[i]

                    cand_cost = route_cost(cand_a, depot, coords) + route_cost(
                        cand_b, depot, coords
                    )
                    delta = cand_cost - base_cost
                    if delta < best_delta:
                        candidate = copy_routes(routes)
                        candidate[ra] = cand_a
                        candidate[rb] = cand_b
                        best_delta = delta
                        best_routes = _cleanup_empty_routes(candidate)

    return best_delta, best_routes


def _best_cross_move(routes, depot, coords, demand, capacity, lam):
    best_delta = 0.0
    best_routes = routes

    for ra in range(len(routes) - 1):
        for rb in range(ra + 1, len(routes)):
            route_a = routes[ra]
            route_b = routes[rb]
            base_cost = route_cost(route_a, depot, coords) + route_cost(
                route_b, depot, coords
            )

            for i in range(len(route_a) + 1):
                for len_a in range(0, lam + 1):
                    if i + len_a > len(route_a):
                        continue
                    seg_a = route_a[i : i + len_a]
                    rem_a = route_a[:i] + route_a[i + len_a :]

                    for j in range(len(route_b) + 1):
                        for len_b in range(0, lam + 1):
                            if j + len_b > len(route_b):
                                continue
                            if len_a == 0 and len_b == 0:
                                continue

                            seg_b = route_b[j : j + len_b]
                            rem_b = route_b[:j] + route_b[j + len_b :]

                            cand_a = rem_a[:i] + seg_b + rem_a[i:]
                            cand_b = rem_b[:j] + seg_a + rem_b[j:]

                            if _route_load(cand_a, demand) > capacity:
                                continue
                            if _route_load(cand_b, demand) > capacity:
                                continue

                            cand_cost = route_cost(cand_a, depot, coords) + route_cost(
                                cand_b, depot, coords
                            )
                            delta = cand_cost - base_cost
                            if delta < best_delta:
                                candidate = copy_routes(routes)
                                candidate[ra] = cand_a
                                candidate[rb] = cand_b
                                best_delta = delta
                                best_routes = _cleanup_empty_routes(candidate)

    return best_delta, best_routes


def _subset_index_tuples(n, lam):
    out = [()]
    for size in range(1, min(lam, n) + 1):
        out.extend(itertools.combinations(range(n), size))
    return out


def _remove_by_indices(route, index_tuple):
    index_set = set(index_tuple)
    picked = [node for idx, node in enumerate(route) if idx in index_set]
    remain = [node for idx, node in enumerate(route) if idx not in index_set]
    return picked, remain


def _best_lambda_interchange_move(
    routes,
    depot,
    coords,
    demand,
    capacity,
    lam,
    max_route_size,
    max_pairs_per_routes,
):
    best_delta = 0.0
    best_routes = routes

    for ra in range(len(routes) - 1):
        for rb in range(ra + 1, len(routes)):
            route_a = routes[ra]
            route_b = routes[rb]
            if len(route_a) > max_route_size or len(route_b) > max_route_size:
                continue

            base_cost = route_cost(route_a, depot, coords) + route_cost(
                route_b, depot, coords
            )
            subsets_a = _subset_index_tuples(len(route_a), lam)
            subsets_b = _subset_index_tuples(len(route_b), lam)

            tested = 0
            for idx_a in subsets_a:
                for idx_b in subsets_b:
                    if not idx_a and not idx_b:
                        continue
                    tested += 1
                    if tested > max_pairs_per_routes:
                        break

                    picked_a, rem_a = _remove_by_indices(route_a, idx_a)
                    picked_b, rem_b = _remove_by_indices(route_b, idx_b)
                    if len(picked_a) > lam or len(picked_b) > lam:
                        continue

                    for rev_a in (False, True):
                        ins_a = list(reversed(picked_b)) if rev_a else picked_b
                        for rev_b in (False, True):
                            ins_b = list(reversed(picked_a)) if rev_b else picked_a

                            for pos_a in range(len(rem_a) + 1):
                                cand_a = rem_a[:pos_a] + ins_a + rem_a[pos_a:]
                                if _route_load(cand_a, demand) > capacity:
                                    continue

                                for pos_b in range(len(rem_b) + 1):
                                    cand_b = rem_b[:pos_b] + ins_b + rem_b[pos_b:]
                                    if _route_load(cand_b, demand) > capacity:
                                        continue

                                    cand_cost = route_cost(
                                        cand_a, depot, coords
                                    ) + route_cost(cand_b, depot, coords)
                                    delta = cand_cost - base_cost
                                    if delta < best_delta:
                                        candidate = copy_routes(routes)
                                        candidate[ra] = cand_a
                                        candidate[rb] = cand_b
                                        best_delta = delta
                                        best_routes = _cleanup_empty_routes(candidate)
                if tested > max_pairs_per_routes:
                    break

    return best_delta, best_routes


def two_opt_star(instance, routes, max_iterations=20):
    depot, coords, demand, capacity = _build_maps(instance)
    return _iterate_solution_improvement(
        routes,
        lambda sol: _best_two_opt_star_move(sol, depot, coords, demand, capacity),
        max_iterations,
    )


def insert(instance, routes, max_iterations=30):
    depot, coords, demand, capacity = _build_maps(instance)
    return _iterate_solution_improvement(
        routes,
        lambda sol: _best_insert_move(sol, depot, coords, demand, capacity),
        max_iterations,
    )


def swap(instance, routes, max_iterations=30):
    depot, coords, demand, capacity = _build_maps(instance)
    return _iterate_solution_improvement(
        routes,
        lambda sol: _best_swap_move(sol, depot, coords, demand, capacity),
        max_iterations,
    )


def cross(instance, routes, lam=3, max_iterations=20):
    depot, coords, demand, capacity = _build_maps(instance)
    return _iterate_solution_improvement(
        routes,
        lambda sol: _best_cross_move(sol, depot, coords, demand, capacity, lam),
        max_iterations,
    )


def lambda_interchange(
    instance,
    routes,
    lam=3,
    max_iterations=10,
    max_route_size=20,
    max_pairs_per_routes=2500,
):
    depot, coords, demand, capacity = _build_maps(instance)
    return _iterate_solution_improvement(
        routes,
        lambda sol: _best_lambda_interchange_move(
            sol,
            depot,
            coords,
            demand,
            capacity,
            lam,
            max_route_size,
            max_pairs_per_routes,
        ),
        max_iterations,
    )


def improve_inter(
    instance,
    routes,
    methods=("2opt*", "insert", "swap", "cross"),
    max_passes=2,
    lam=3,
):
    """
    Apply a sequence of inter-route heuristics to the full solution.
    """
    method_map = {
        "2opt*": lambda inst, rts: two_opt_star(inst, rts),
        "insert": lambda inst, rts: insert(inst, rts),
        "swap": lambda inst, rts: swap(inst, rts),
        "cross": lambda inst, rts: cross(inst, rts, lam=lam),
        "lambda_interchange": lambda inst, rts: lambda_interchange(inst, rts, lam=lam),
    }

    current = copy_routes(routes)
    current_cost = solution_cost(instance, current)

    for _ in range(max_passes):
        improved_in_pass = False
        for name in methods:
            if name not in method_map:
                raise ValueError(f"Unknown inter-route method: {name}")
            candidate = method_map[name](instance, current)
            candidate_cost = solution_cost(instance, candidate)
            if candidate_cost + 1e-9 < current_cost:
                current = candidate
                current_cost = candidate_cost
                improved_in_pass = True
        if not improved_in_pass:
            break

    return current


__all__ = [
    "two_opt_star",
    "insert",
    "swap",
    "cross",
    "lambda_interchange",
    "improve_inter",
]
