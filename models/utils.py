"""
Shared helper utilities for CVRP models.
"""

import math
import numpy as np


def extract_instance_data(instance):
    """Support both dict-based instances and tsplib95 StandardProblem objects."""
    if isinstance(instance, dict):
        depot = instance["depot"]
        customers = instance["customers"]
        if isinstance(customers, dict):
            customer_ids = list(customers.keys())
            customers = [customers[cid] for cid in customer_ids]
            demands = instance["demands"]
            if isinstance(demands, dict):
                demands = [demands[cid] for cid in customer_ids]
            else:
                demands = list(demands)
        else:
            customer_ids = list(range(len(customers)))
            demands = list(instance["demands"])

        capacity = instance["capacity"]
        return depot, customers, demands, capacity, customer_ids

    if hasattr(instance, "node_coords") and hasattr(instance, "demands") and hasattr(instance, "capacity"):
        depots = list(getattr(instance, "depots", []))
        depot_id = depots[0] if depots else 1
        node_coords = instance.node_coords
        customer_ids = sorted(node for node in node_coords if node != depot_id)
        depot = node_coords[depot_id]
        customers = [node_coords[node] for node in customer_ids]
        demands = [instance.demands.get(node, 0) for node in customer_ids]
        capacity = instance.capacity
        return depot, customers, demands, capacity, customer_ids

    raise TypeError("Unsupported instance format. Use dict or tsplib95 problem object.")


def extract_geometry(instance):
    """
    Return:
    - depot: coordinate tuple
    - customer_coords: {customer_id: (x, y)}
    """
    depot, customers, _, _, customer_ids = extract_instance_data(instance)
    customer_coords = {customer_ids[idx]: coord for idx, coord in enumerate(customers)}
    return depot, customer_coords


def validate_demands_capacity(demands, capacity):
    if any(d > capacity for d in demands):
        raise ValueError("At least one customer demand exceeds vehicle capacity.")


def decode_routes(routes, customer_ids):
    return [[customer_ids[c] for c in route] for route in routes]


def copy_routes(routes):
    return [list(route) for route in routes]


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_cost(route, depot, customer_coords):
    if not route:
        return 0.0
    cost = euclidean(depot, customer_coords[route[0]])
    for i in range(len(route) - 1):
        cost += euclidean(customer_coords[route[i]], customer_coords[route[i + 1]])
    cost += euclidean(customer_coords[route[-1]], depot)
    return cost


def solution_cost(instance, routes):
    depot, customer_coords = extract_geometry(instance)
    return sum(route_cost(route, depot, customer_coords) for route in routes)


def calculate_route_cost(route, depot, customers):
    """Route format may include -1 as depot marker at first position."""
    customers_only = [c for c in route if c != -1]
    if not customers_only:
        return 0.0

    cost = np.linalg.norm(np.array(depot) - np.array(customers[customers_only[0]]))
    for i in range(len(customers_only) - 1):
        a = customers[customers_only[i]]
        b = customers[customers_only[i + 1]]
        cost += np.linalg.norm(np.array(a) - np.array(b))
    cost += np.linalg.norm(np.array(customers[customers_only[-1]]) - np.array(depot))
    return float(cost)


__all__ = [
    "extract_instance_data",
    "extract_geometry",
    "validate_demands_capacity",
    "decode_routes",
    "copy_routes",
    "euclidean",
    "route_cost",
    "solution_cost",
    "calculate_route_cost",
]
