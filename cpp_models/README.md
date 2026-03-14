# C++ CVRP Model Programs

This folder contains C++ program files that mirror the Python models in `models/`.

## Build

```bash
cmake -S cpp_models -B cpp_models/build
cmake --build cpp_models/build -j
```

If `cmake` is not installed, `models/cpp_bridge.py::build_cpp_models()` can build via `g++` directly.

## Produced executables

- `cvrp_constructive`
- `cvrp_improve_intra`
- `cvrp_improve_inter`
- `cvrp_tabu`
- `cvrp_simulated_annealing`
- `cvrp_iterated_local_search`
- `cvrp_large_neighborhood_search`
- `cvrp_genetic`
- `cvrp_memetic`
- `cvrp_ant_colony`

## CLI basics

All executables support at least:

- `--instance <path>`: required
- `--routes <path>`: optional initial solution (for models that accept it)

Instance file format:

```text
capacity <number>
depot <x> <y>
customers <N>
<id> <x> <y> <demand>
...
```

Routes file format (one route per line):

```text
1 7 2 5
3 6 4
...
```

Each executable prints one JSON object to stdout with at least:

- `status`
- `best_cost`
- `best_routes`
- `final_cost`
- `final_routes`
- `iterations`

For Python usage, prefer `models/cpp_bridge.py`.
