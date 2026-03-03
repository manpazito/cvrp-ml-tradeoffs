"""
Tabu Search for CVRP.

Core ideas:
- Keep a tabu list to avoid cycling.
- Explore sampled neighbors each iteration.
- Accept the best admissible neighbor (or tabu neighbor via aspiration).
- Track the global best solution.
"""

import random

try:
    from models.constructive import nearest_neighbor
    from models.utils import copy_routes, extract_instance_data, solution_cost
except ImportError:
    from constructive import nearest_neighbor
    from utils import copy_routes, extract_instance_data, solution_cost


def _build_maps(instance):
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    demand = {customer_ids[i]: demands[i] for i in range(len(customer_ids))}
    customer_set = set(customer_ids)
    return depot, demand, capacity, customer_set


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


def _solution_signature(routes):
    # Route order is not meaningful for CVRP, so sort route tuples for tabu hashing.
    return tuple(sorted(tuple(route) for route in routes))


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


def _move_intra_swap(routes, rng):
    new_routes = copy_routes(routes)
    ridx = _pick_route_index(new_routes, rng, min_size=2)
    if ridx is None:
        return None
    route = new_routes[ridx]
    i, j = rng.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
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


def _move_cross(routes, rng, lam):
    new_routes = copy_routes(routes)
    a = _pick_route_index(new_routes, rng, min_size=0)
    if a is None:
        return None
    b = _pick_route_index(new_routes, rng, min_size=0, different_from=a)
    if b is None:
        return None
    ra = new_routes[a]
    rb = new_routes[b]

    max_a = min(lam, len(ra))
    max_b = min(lam, len(rb))
    len_a = rng.randint(0, max_a)
    len_b = rng.randint(0, max_b)
    if len_a == 0 and len_b == 0:
        return None

    start_a = rng.randint(0, len(ra) - len_a) if len_a > 0 else rng.randint(0, len(ra))
    start_b = rng.randint(0, len(rb) - len_b) if len_b > 0 else rng.randint(0, len(rb))

    seg_a = ra[start_a : start_a + len_a]
    seg_b = rb[start_b : start_b + len_b]
    rem_a = ra[:start_a] + ra[start_a + len_a :]
    rem_b = rb[:start_b] + rb[start_b + len_b :]

    new_routes[a] = rem_a[:start_a] + seg_b + rem_a[start_a:]
    new_routes[b] = rem_b[:start_b] + seg_a + rem_b[start_b:]
    return _cleanup(new_routes)


def _move_lambda_interchange(routes, rng, lam):
    new_routes = copy_routes(routes)
    a = _pick_route_index(new_routes, rng, min_size=1)
    if a is None:
        return None
    b = _pick_route_index(new_routes, rng, min_size=1, different_from=a)
    if b is None:
        return None
    ra = new_routes[a]
    rb = new_routes[b]

    size_a = rng.randint(0, min(lam, len(ra)))
    size_b = rng.randint(0, min(lam, len(rb)))
    if size_a == 0 and size_b == 0:
        return None

    idx_a = sorted(rng.sample(range(len(ra)), size_a)) if size_a > 0 else []
    idx_b = sorted(rng.sample(range(len(rb)), size_b)) if size_b > 0 else []
    set_a = set(idx_a)
    set_b = set(idx_b)

    take_a = [ra[i] for i in idx_a]
    take_b = [rb[i] for i in idx_b]
    rem_a = [ra[i] for i in range(len(ra)) if i not in set_a]
    rem_b = [rb[i] for i in range(len(rb)) if i not in set_b]

    if rng.random() < 0.5:
        take_a = list(reversed(take_a))
    if rng.random() < 0.5:
        take_b = list(reversed(take_b))

    pos_a = rng.randrange(len(rem_a) + 1)
    pos_b = rng.randrange(len(rem_b) + 1)
    new_routes[a] = rem_a[:pos_a] + take_b + rem_a[pos_a:]
    new_routes[b] = rem_b[:pos_b] + take_a + rem_b[pos_b:]
    return _cleanup(new_routes)


def _neighbor_factory(lam):
    return {
        "intra_2opt": lambda routes, rng: _move_intra_2opt(routes, rng),
        "intra_relocate": lambda routes, rng: _move_intra_relocate(routes, rng),
        "intra_swap": lambda routes, rng: _move_intra_swap(routes, rng),
        "inter_insert": lambda routes, rng: _move_inter_insert(routes, rng),
        "inter_swap": lambda routes, rng: _move_inter_swap(routes, rng),
        "two_opt_star": lambda routes, rng: _move_two_opt_star(routes, rng),
        "cross": lambda routes, rng: _move_cross(routes, rng, lam),
        "lambda_interchange": lambda routes, rng: _move_lambda_interchange(routes, rng, lam),
    }


def _kick_solution(routes, move_map, moves, rng, demand, capacity, customer_set, kick_strength):
    kicked = copy_routes(routes)
    for _ in range(kick_strength):
        move_name = rng.choice(moves)
        candidate = move_map[move_name](kicked, rng)
        if candidate is None:
            continue
        if _feasible_solution(candidate, demand, capacity, customer_set):
            kicked = candidate
    return kicked


def tabu_search(
    instance,
    initial_routes=None,
    max_iterations=5000,
    tabu_tenure=35,
    neighborhood_samples=120,
    seed=42,
    moves=(
        "intra_2opt",
        "intra_relocate",
        "intra_swap",
        "inter_insert",
        "inter_swap",
        "two_opt_star",
        "cross",
        "lambda_interchange",
    ),
    lam=3,
    aspiration=True,
    diversification=True,
    diversification_limit=400,
    kick_strength=4,
    return_history=True,
):
    """
    Tabu Search for CVRP minimization.

    - Neighborhood is sampled randomly at each iteration.
    - Tabu stores recently visited solution signatures.
    - Aspiration allows tabu moves if they improve global best.
    """
    rng = random.Random(seed)
    _, demand, capacity, customer_set = _build_maps(instance)

    if initial_routes is None:
        current_routes = nearest_neighbor(instance)
    else:
        current_routes = copy_routes(initial_routes)

    if not _feasible_solution(current_routes, demand, capacity, customer_set):
        raise ValueError("Initial routes are not feasible for this CVRP instance.")

    move_map = _neighbor_factory(lam)
    unknown_moves = [name for name in moves if name not in move_map]
    if unknown_moves:
        raise ValueError("Unknown move(s): " + ", ".join(unknown_moves))

    current_cost = solution_cost(instance, current_routes)
    best_routes = copy_routes(current_routes)
    best_cost = current_cost

    tabu_until = {}
    iterations = 0
    no_improve = 0
    accepted = 0
    history = []

    while iterations < max_iterations:
        iterations += 1

        best_candidate = None
        best_candidate_cost = float("inf")
        best_candidate_sig = None

        # Sample neighborhood and pick best admissible move
        for _ in range(neighborhood_samples):
            move_name = rng.choice(moves)
            candidate = move_map[move_name](current_routes, rng)
            if candidate is None:
                continue
            if not _feasible_solution(candidate, demand, capacity, customer_set):
                continue

            cand_cost = solution_cost(instance, candidate)
            cand_sig = _solution_signature(candidate)
            is_tabu = tabu_until.get(cand_sig, 0) > iterations

            if is_tabu and aspiration and cand_cost < best_cost - 1e-9:
                pass
            elif is_tabu:
                continue

            if cand_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = cand_cost
                best_candidate_sig = cand_sig

        if best_candidate is None:
            # If no admissible candidate found, diversify or stop.
            if diversification:
                current_routes = _kick_solution(
                    current_routes,
                    move_map,
                    moves,
                    rng,
                    demand,
                    capacity,
                    customer_set,
                    kick_strength,
                )
                current_cost = solution_cost(instance, current_routes)
                no_improve += 1
                continue
            break

        current_routes = best_candidate
        current_cost = best_candidate_cost
        accepted += 1

        tenure = tabu_tenure + rng.randint(0, max(1, tabu_tenure // 4))
        tabu_until[best_candidate_sig] = iterations + tenure

        if current_cost < best_cost - 1e-9:
            best_cost = current_cost
            best_routes = copy_routes(current_routes)
            no_improve = 0
        else:
            no_improve += 1

        if diversification and no_improve >= diversification_limit:
            current_routes = _kick_solution(
                current_routes,
                move_map,
                moves,
                rng,
                demand,
                capacity,
                customer_set,
                kick_strength,
            )
            current_cost = solution_cost(instance, current_routes)
            no_improve = 0

        if iterations % 100 == 0:
            # Light cleanup of expired tabu entries.
            expired = [sig for sig, until in tabu_until.items() if until <= iterations]
            for sig in expired:
                del tabu_until[sig]

        if return_history and iterations % 10 == 0:
            history.append(
                {
                    "iteration": iterations,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "tabu_size": len(tabu_until),
                    "accepted": accepted,
                }
            )

    return {
        "best_routes": best_routes,
        "best_cost": best_cost,
        "final_routes": current_routes,
        "final_cost": current_cost,
        "iterations": iterations,
        "accepted_moves": accepted,
        "history": history if return_history else None,
    }


def tabu_from_nearest_neighbor(instance, **kwargs):
    """
    Convenience wrapper: builds initial solution with nearest neighbor and runs tabu search.
    """
    init_routes = nearest_neighbor(instance)
    return tabu_search(instance, initial_routes=init_routes, **kwargs)


__all__ = ["tabu_search", "tabu_from_nearest_neighbor"]
