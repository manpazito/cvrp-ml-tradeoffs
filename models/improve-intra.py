"""
Intra-route improvement heuristics for CVRP.

Includes:
- 2-opt
- 3-opt
- Or-opt
- Relocate
- Exchange
- GENI-inspired reinsertion
"""

try:
    from models.utils import (
        copy_routes,
        euclidean,
        extract_geometry,
        route_cost,
        solution_cost,
    )
except ImportError:
    from utils import copy_routes, euclidean, extract_geometry, route_cost, solution_cost


def _best_two_opt_move(route, depot, customer_coords):
    n = len(route)
    if n < 4:
        return 0.0, route

    base_cost = route_cost(route, depot, customer_coords)
    best_delta = 0.0
    best_route = route

    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            candidate = route[: i + 1] + list(reversed(route[i + 1 : j + 1])) + route[j + 1 :]
            candidate_cost = route_cost(candidate, depot, customer_coords)
            delta = candidate_cost - base_cost
            if delta < best_delta:
                best_delta = delta
                best_route = candidate

    return best_delta, best_route


def _best_three_opt_move(route, depot, customer_coords):
    n = len(route)
    if n < 6:
        return 0.0, route

    base_cost = route_cost(route, depot, customer_coords)
    best_delta = 0.0
    best_route = route

    for i in range(n - 5):
        for j in range(i + 2, n - 3):
            for k in range(j + 2, n - 1):
                a = route[: i + 1]
                b = route[i + 1 : j + 1]
                c = route[j + 1 : k + 1]
                d = route[k + 1 :]

                candidates = [
                    a + list(reversed(b)) + c + d,
                    a + b + list(reversed(c)) + d,
                    a + list(reversed(b)) + list(reversed(c)) + d,
                    a + c + b + d,
                    a + list(reversed(c)) + b + d,
                    a + c + list(reversed(b)) + d,
                    a + list(reversed(c)) + list(reversed(b)) + d,
                ]

                for candidate in candidates:
                    candidate_cost = route_cost(candidate, depot, customer_coords)
                    delta = candidate_cost - base_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_route = candidate

    return best_delta, best_route


def _best_or_opt_move(route, depot, customer_coords, segment_lengths):
    n = len(route)
    if n < 3:
        return 0.0, route

    base_cost = route_cost(route, depot, customer_coords)
    best_delta = 0.0
    best_route = route

    for seg_len in segment_lengths:
        if seg_len <= 0 or seg_len >= n:
            continue
        for i in range(n - seg_len + 1):
            segment = route[i : i + seg_len]
            remainder = route[:i] + route[i + seg_len :]
            for j in range(len(remainder) + 1):
                if j == i:
                    continue
                candidate = remainder[:j] + segment + remainder[j:]
                candidate_cost = route_cost(candidate, depot, customer_coords)
                delta = candidate_cost - base_cost
                if delta < best_delta:
                    best_delta = delta
                    best_route = candidate

    return best_delta, best_route


def _best_relocate_move(route, depot, customer_coords):
    return _best_or_opt_move(route, depot, customer_coords, (1,))


def _best_exchange_move(route, depot, customer_coords):
    n = len(route)
    if n < 2:
        return 0.0, route

    base_cost = route_cost(route, depot, customer_coords)
    best_delta = 0.0
    best_route = route

    for i in range(n - 1):
        for j in range(i + 1, n):
            candidate = list(route)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            candidate_cost = route_cost(candidate, depot, customer_coords)
            delta = candidate_cost - base_cost
            if delta < best_delta:
                best_delta = delta
                best_route = candidate

    return best_delta, best_route


def _nearest_nodes(node, pool, customer_coords, depot, k):
    if node not in customer_coords or k <= 0:
        return set()
    node_coord = customer_coords[node]
    scored = []
    for other in pool:
        if other == node:
            continue
        scored.append((euclidean(node_coord, customer_coords[other]), other))
    scored.sort(key=lambda item: item[0])
    return {other for _, other in scored[:k]}


def _best_geni_move(route, depot, customer_coords, nearest_k):
    """
    GENI-inspired move:
    remove a customer and reinsert it near one of its nearest neighbors.
    """
    n = len(route)
    if n < 3:
        return 0.0, route

    base_cost = route_cost(route, depot, customer_coords)
    best_delta = 0.0
    best_route = route

    for i, node in enumerate(route):
        reduced = route[:i] + route[i + 1 :]
        nearest = _nearest_nodes(node, reduced, customer_coords, depot, nearest_k)

        for j in range(len(reduced) + 1):
            prev_node = reduced[j - 1] if j > 0 else None
            next_node = reduced[j] if j < len(reduced) else None

            if nearest:
                if (prev_node is not None and prev_node not in nearest) and (
                    next_node is not None and next_node not in nearest
                ):
                    continue

            candidate = reduced[:j] + [node] + reduced[j:]
            candidate_cost = route_cost(candidate, depot, customer_coords)
            delta = candidate_cost - base_cost
            if delta < best_delta:
                best_delta = delta
                best_route = candidate

    return best_delta, best_route


def _iterate_route_improvement(route, move_fn, max_iterations):
    current = list(route)
    for _ in range(max_iterations):
        delta, candidate = move_fn(current)
        if delta < -1e-9:
            current = candidate
        else:
            break
    return current


def _apply_to_routes(instance, routes, route_move_fn, max_iterations):
    depot, customer_coords = extract_geometry(instance)
    improved = copy_routes(routes)
    for idx, route in enumerate(improved):
        improved[idx] = _iterate_route_improvement(
            route,
            lambda r: route_move_fn(r, depot, customer_coords),
            max_iterations,
        )
    return improved


def two_opt(instance, routes, max_iterations=30):
    return _apply_to_routes(instance, routes, _best_two_opt_move, max_iterations)


def three_opt(instance, routes, max_iterations=10, max_route_size=120):
    depot, customer_coords = extract_geometry(instance)
    improved = copy_routes(routes)
    for idx, route in enumerate(improved):
        if len(route) > max_route_size:
            continue
        improved[idx] = _iterate_route_improvement(
            route,
            lambda r: _best_three_opt_move(r, depot, customer_coords),
            max_iterations,
        )
    return improved


def or_opt(instance, routes, segment_lengths=(1, 2, 3), max_iterations=30):
    return _apply_to_routes(
        instance,
        routes,
        lambda route, depot, customer_coords: _best_or_opt_move(
            route, depot, customer_coords, segment_lengths
        ),
        max_iterations,
    )


def relocate(instance, routes, max_iterations=30):
    return _apply_to_routes(instance, routes, _best_relocate_move, max_iterations)


def exchange(instance, routes, max_iterations=30):
    return _apply_to_routes(instance, routes, _best_exchange_move, max_iterations)


def geni(instance, routes, nearest_k=8, max_iterations=30):
    return _apply_to_routes(
        instance,
        routes,
        lambda route, depot, customer_coords: _best_geni_move(
            route, depot, customer_coords, nearest_k
        ),
        max_iterations,
    )


def improve_intra(
    instance,
    routes,
    methods=("2opt", "or_opt", "relocate", "exchange", "geni"),
    max_passes=2,
):
    """
    Apply a sequence of intra-route heuristics to the full solution.
    """
    method_map = {
        "2opt": two_opt,
        "3opt": three_opt,
        "or_opt": or_opt,
        "relocate": relocate,
        "exchange": exchange,
        "geni": geni,
    }

    current = copy_routes(routes)
    current_cost = solution_cost(instance, current)

    for _ in range(max_passes):
        improved_in_pass = False
        for method_name in methods:
            if method_name not in method_map:
                raise ValueError(f"Unknown improvement method: {method_name}")
            candidate = method_map[method_name](instance, current)
            candidate_cost = solution_cost(instance, candidate)
            if candidate_cost + 1e-9 < current_cost:
                current = candidate
                current_cost = candidate_cost
                improved_in_pass = True
        if not improved_in_pass:
            break

    return current


__all__ = [
    "route_cost",
    "solution_cost",
    "two_opt",
    "three_opt",
    "or_opt",
    "relocate",
    "exchange",
    "geni",
    "improve_intra",
]
