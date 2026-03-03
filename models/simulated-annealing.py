"""
Simulated Annealing for CVRP.

The algorithm:
1) Starts from an initial feasible solution.
2) Generates random neighboring solutions using local operators.
3) Accepts improving moves, and sometimes worse moves with probability exp(-delta / T).
4) Decreases temperature until stopping criteria is reached.
"""

import math
import random

try:
    from models.constructive import nearest_neighbor
    from models.utils import copy_routes, extract_instance_data, solution_cost
except ImportError:
    from constructive import nearest_neighbor
    from utils import copy_routes, extract_instance_data, solution_cost


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


def _pick_route_indices(routes, rng, min_size=1, different_from=None):
    valid = [idx for idx, route in enumerate(routes) if len(route) >= min_size]
    if different_from is not None:
        valid = [idx for idx in valid if idx != different_from]
    if not valid:
        return None
    return rng.choice(valid)


def _move_intra_2opt(routes, rng):
    new_routes = copy_routes(routes)
    ridx = _pick_route_indices(new_routes, rng, min_size=4)
    if ridx is None:
        return None
    route = new_routes[ridx]
    i = rng.randint(0, len(route) - 3)
    j = rng.randint(i + 2, len(route) - 1)
    route[i : j + 1] = reversed(route[i : j + 1])
    return _cleanup(new_routes)


def _move_intra_relocate(routes, rng):
    new_routes = copy_routes(routes)
    ridx = _pick_route_indices(new_routes, rng, min_size=2)
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
    ridx = _pick_route_indices(new_routes, rng, min_size=2)
    if ridx is None:
        return None
    route = new_routes[ridx]
    i, j = rng.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return _cleanup(new_routes)


def _move_inter_insert(routes, rng):
    new_routes = copy_routes(routes)
    src = _pick_route_indices(new_routes, rng, min_size=1)
    if src is None:
        return None
    dst = _pick_route_indices(new_routes, rng, min_size=0, different_from=src)
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
    a = _pick_route_indices(new_routes, rng, min_size=1)
    if a is None:
        return None
    b = _pick_route_indices(new_routes, rng, min_size=1, different_from=a)
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
    a = _pick_route_indices(new_routes, rng, min_size=1)
    if a is None:
        return None
    b = _pick_route_indices(new_routes, rng, min_size=1, different_from=a)
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
    a = _pick_route_indices(new_routes, rng, min_size=0)
    if a is None:
        return None
    b = _pick_route_indices(new_routes, rng, min_size=0, different_from=a)
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
    a = _pick_route_indices(new_routes, rng, min_size=1)
    if a is None:
        return None
    b = _pick_route_indices(new_routes, rng, min_size=1, different_from=a)
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


def _choose_move(moves, weights, rng):
    if not weights or len(weights) != len(moves):
        return rng.choice(moves)
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return rng.choice(moves)
    ticket = rng.random() * total
    acc = 0.0
    for name, weight in zip(moves, weights):
        acc += max(0.0, weight)
        if ticket <= acc:
            return name
    return moves[-1]


def _default_initial_temperature(cost):
    return max(1.0, 0.05 * max(1.0, cost))


def simulated_annealing(
    instance,
    initial_routes=None,
    initial_temperature=None,
    final_temperature=1e-3,
    cooling_rate=0.995,
    iterations_per_temperature=200,
    max_iterations=50000,
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
    max_neighbor_attempts=50,
    adaptive_moves=True,
    reheat=False,
    stagnation_limit=2500,
    reheat_factor=1.5,
    return_history=True,
):
    """
    Simulated Annealing for CVRP minimization.
    """
    rng = random.Random(seed)
    depot, coords, demand, capacity, customer_set = _build_maps(instance)

    if initial_routes is None:
        current_routes = nearest_neighbor(instance)
    else:
        current_routes = copy_routes(initial_routes)

    if not _feasible_solution(current_routes, demand, capacity, customer_set):
        raise ValueError("Initial routes are not feasible for this CVRP instance.")

    current_cost = solution_cost(instance, current_routes)
    best_routes = copy_routes(current_routes)
    best_cost = current_cost

    temperature = (
        _default_initial_temperature(current_cost)
        if initial_temperature is None
        else float(initial_temperature)
    )
    if temperature <= 0:
        raise ValueError("initial_temperature must be > 0")

    move_map = _neighbor_factory(lam)
    unknown = [move for move in moves if move not in move_map]
    if unknown:
        raise ValueError("Unknown move(s): " + ", ".join(unknown))

    move_weights = [1.0] * len(moves)
    accepted = 0
    improving = 0
    rejected = 0
    iterations = 0
    no_improve_steps = 0
    history = []

    while (
        temperature > final_temperature
        and iterations < max_iterations
    ):
        for _ in range(iterations_per_temperature):
            if iterations >= max_iterations:
                break
            iterations += 1

            neighbor_routes = None
            chosen_idx = None
            for _attempt in range(max_neighbor_attempts):
                move_name = _choose_move(moves, move_weights, rng)
                chosen_idx = moves.index(move_name)
                candidate = move_map[move_name](current_routes, rng)
                if candidate is None:
                    continue
                if not _feasible_solution(candidate, demand, capacity, customer_set):
                    continue
                neighbor_routes = candidate
                break

            if neighbor_routes is None:
                rejected += 1
                continue

            neighbor_cost = solution_cost(instance, neighbor_routes)
            delta = neighbor_cost - current_cost

            accept = False
            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / max(temperature, 1e-12))
                if rng.random() < prob:
                    accept = True

            if accept:
                current_routes = neighbor_routes
                current_cost = neighbor_cost
                accepted += 1
                if delta < 0:
                    improving += 1
                    no_improve_steps = 0
                    if adaptive_moves and chosen_idx is not None:
                        move_weights[chosen_idx] *= 1.03
                else:
                    no_improve_steps += 1
                    if adaptive_moves and chosen_idx is not None:
                        move_weights[chosen_idx] *= 0.999

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_routes = copy_routes(current_routes)
                    no_improve_steps = 0
            else:
                rejected += 1
                no_improve_steps += 1
                if adaptive_moves and chosen_idx is not None:
                    move_weights[chosen_idx] *= 0.997

        if return_history:
            history.append(
                {
                    "temperature": temperature,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "accepted": accepted,
                    "improving": improving,
                    "rejected": rejected,
                    "iterations": iterations,
                }
            )

        if reheat and no_improve_steps >= stagnation_limit:
            temperature *= reheat_factor
            no_improve_steps = 0
        else:
            temperature *= cooling_rate

    return {
        "best_routes": best_routes,
        "best_cost": best_cost,
        "final_routes": current_routes,
        "final_cost": current_cost,
        "iterations": iterations,
        "accepted_moves": accepted,
        "improving_moves": improving,
        "rejected_moves": rejected,
        "history": history if return_history else None,
    }


def anneal_from_nearest_neighbor(instance, **kwargs):
    """
    Convenience wrapper: builds initial solution with nearest neighbor and runs SA.
    """
    init_routes = nearest_neighbor(instance)
    return simulated_annealing(instance, initial_routes=init_routes, **kwargs)


__all__ = ["simulated_annealing", "anneal_from_nearest_neighbor"]
