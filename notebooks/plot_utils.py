import matplotlib.pyplot as plt
import seaborn as sns


def plot_cvrp_routes(problem, routes, instance_id=None):
    """
    Plot CVRP routes with each route in a different color.

    Parameters
    ----------
    problem : tsplib95.models.StandardProblem
        Loaded CVRP instance.
    routes : list[list[int]]
        List of vehicle routes (each route is a list of node IDs).
    instance_id : str, optional
        Instance name for plot title.
    """

    coords = getattr(problem, "node_coords", {}) or {}
    depot_ids = list(getattr(problem, "depots", []) or [])

    if not depot_ids and coords:
        depot_ids = [min(coords)]

    if not coords or not depot_ids:
        raise ValueError("Problem missing coordinates or depot information.")

    depot_id = depot_ids[0]
    depot_x, depot_y = coords[depot_id]

    plt.figure(figsize=(6, 6))

    # Depot
    plt.scatter(depot_x, depot_y, c="red", marker="X", s=120, label="Depot")

    # Customers
    customer_ids = [nid for nid in coords if nid != depot_id]
    xs = [coords[n][0] for n in customer_ids]
    ys = [coords[n][1] for n in customer_ids]
    plt.scatter(xs, ys, color="gray", s=20)

    # Route colors
    palette = sns.color_palette("tab20", len(routes))

    # Plot each route
    for i, route in enumerate(routes):

        route_x = [depot_x]
        route_y = [depot_y]

        for node in route:
            x, y = coords[node]
            route_x.append(x)
            route_y.append(y)

        route_x.append(depot_x)
        route_y.append(depot_y)

        plt.plot(
            route_x,
            route_y,
            color=palette[i],
            linewidth=2,
            marker="o",
            label=f"Route {i+1}",
        )

    if instance_id:
        plt.title(f"CVRP Routes: {instance_id}")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
