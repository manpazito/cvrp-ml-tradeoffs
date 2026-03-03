"""
Constructive Heuristics for the Capacitated Vehicle Routing Problem (CVRP).
"""
import numpy as np

try:
    from models.utils import (
        calculate_route_cost,
        decode_routes,
        extract_instance_data,
        validate_demands_capacity,
    )
except ImportError:
    from utils import (
        calculate_route_cost,
        decode_routes,
        extract_instance_data,
        validate_demands_capacity,
    )

####################################
#### Nearest Neighbor Heuristic ####
####################################

def nearest_neighbor(instance):
    """
    Construct a CVRP solution using the Nearest Neighbor heuristic.
    
    Parameters:
    - instance: A CVRP instance containing the depot, customers, demands, and vehicle capacity.
    
    Returns:
    - A list of routes, where each route is a list of customer indices.
    """
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    validate_demands_capacity(demands, capacity)
    
    unvisited = set(range(len(customers)))
    routes = []
    
    while unvisited:
        route = []
        load = 0
        current_location = depot
        
        while True:
            # Find the nearest unvisited customer that can be added to the route
            nearest_customer = None
            nearest_distance = float('inf')
            
            for customer in unvisited:
                if load + demands[customer] <= capacity:
                    distance = np.linalg.norm(np.array(current_location) - np.array(customers[customer]))
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_customer = customer
            
            if nearest_customer is None:
                break
            
            # Add the nearest customer to the route
            route.append(nearest_customer)
            load += demands[nearest_customer]
            current_location = customers[nearest_customer]
            unvisited.remove(nearest_customer)

        if not route:
            raise ValueError("Could not build a feasible route with current capacity/demands.")
        routes.append(route)
    
    return decode_routes(routes, customer_ids)


#########################################
#### Clarke-Wright Savings Heuristic ####
#########################################

def clarke_wright(instance):
    """
    Construct a CVRP solution using the Clarke-Wright Savings heuristic.
    
    Parameters:
    - instance: A CVRP instance containing the depot, customers, demands, and vehicle capacity.
    
    Returns:
    - A list of routes, where each route is a list of customer indices.
    """
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    validate_demands_capacity(demands, capacity)
    
    # Step 1: Start with a route for each customer
    routes = [[i] for i in range(len(customers))]
    
    # Step 2: Calculate savings for merging routes
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            distance_ij = np.linalg.norm(np.array(customers[i]) - np.array(customers[j]))
            distance_id = np.linalg.norm(np.array(depot) - np.array(customers[i]))
            distance_jd = np.linalg.norm(np.array(depot) - np.array(customers[j]))
            saving = distance_id + distance_jd - distance_ij
            savings.append((saving, i, j))
    
    # Step 3: Sort savings in descending order
    savings.sort(reverse=True)
    
    # Step 4: Merge routes based on savings
    for saving, i, j in savings:
        route_i = next((route for route in routes if i in route), None)
        route_j = next((route for route in routes if j in route), None)
        
        if route_i is not None and route_j is not None and route_i != route_j:
            load_i = sum(demands[c] for c in route_i)
            load_j = sum(demands[c] for c in route_j)
            
            # Merge only when i and j are at route ends (standard CW feasibility condition).
            if (
                load_i + load_j <= capacity
                and i in (route_i[0], route_i[-1])
                and j in (route_j[0], route_j[-1])
            ):
                route_i_oriented = list(reversed(route_i)) if route_i[0] == i else list(route_i)
                route_j_oriented = list(reversed(route_j)) if route_j[-1] == j else list(route_j)
                # Merge the two routes
                new_route = route_i_oriented + route_j_oriented
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(new_route)
    
    return decode_routes(routes, customer_ids)


############################################
####### Cheapest Insertion Heuristic #######
############################################

def cheapest_insertion(instance):
    """
    Construct a CVRP solution using the Cheapest Insertion heuristic. Builds routes incrementally.

    Parameters:
    - instance: A CVRP instance containing the depot, customers, demands, and vehicle capacity.

    Returns:
    - A list of routes, where each route is a list of customer indices.
    """
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    validate_demands_capacity(demands, capacity)

    # Start with an initial route containing only the depot
    routes = [[-1]]  # -1 represents the depot

    unvisited = set(range(len(customers)))

    while unvisited:
        best_insertion = None
        best_cost = float('inf')

        for customer in unvisited:
            for route in routes:
                for i in range(1, len(route) + 1):
                    new_route = route[:i] + [customer] + route[i:]
                    load = sum(demands[c] for c in new_route if c != -1)

                    if load <= capacity:
                        cost = calculate_route_cost(new_route, depot, customers)
                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (customer, route, i)

        if best_insertion is not None:
            customer, route, position = best_insertion
            route.insert(position, customer)
            unvisited.remove(customer)
        else:
            # If no feasible insertion is found, start a new route
            routes.append([-1])  # Start a new route with the depot

    result = [route[1:] for route in routes if len(route) > 1]  # Remove depot from routes
    return decode_routes(result, customer_ids)

##########################
### Sweep Heuristic ###
###########################

def sweep(instance):
    """
    Construct a CVRP solution using the Sweep heuristic. Sorts customers by polar angle and builds routes sequentially.

    Parameters:
    - instance: A CVRP instance containing the depot, customers, demands, and vehicle capacity.

    Returns:
    - A list of routes, where each route is a list of customer indices.
    """
    depot, customers, demands, capacity, customer_ids = extract_instance_data(instance)
    validate_demands_capacity(demands, capacity)

    # Calculate polar angles for each customer
    angles = []
    for i, customer in enumerate(customers):
        angle = np.arctan2(customer[1] - depot[1], customer[0] - depot[0])
        angles.append((angle, i))

    # Sort customers by polar angle
    angles.sort()

    routes = []
    current_route = []
    current_load = 0

    for _, customer_index in angles:
        demand = demands[customer_index]
        if current_load + demand <= capacity:
            current_route.append(customer_index)
            current_load += demand
        else:
            routes.append(current_route)
            current_route = [customer_index]
            current_load = demand

    if current_route:
        routes.append(current_route)

    return decode_routes(routes, customer_ids)
