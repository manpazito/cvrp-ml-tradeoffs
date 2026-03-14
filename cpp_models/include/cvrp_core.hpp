#pragma once

#include <limits>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cvrp {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

using CustomerId = int;
using Route = std::vector<CustomerId>;
using Routes = std::vector<Route>;

struct Instance {
    Point depot{};
    double capacity = 0.0;
    std::vector<CustomerId> customer_ids;
    std::unordered_map<CustomerId, Point> coords;
    std::unordered_map<CustomerId, double> demand;
};

struct SolverOutput {
    Routes best_routes;
    double best_cost = std::numeric_limits<double>::infinity();
    Routes final_routes;
    double final_cost = std::numeric_limits<double>::infinity();
    int iterations = 0;
    std::vector<double> history_best_cost;
};

struct IntraImproveOptions {
    std::vector<std::string> methods = {"2opt", "or_opt", "relocate", "exchange", "geni"};
    int max_passes = 2;
    int max_iterations = 30;
    int three_opt_max_route_size = 120;
    int nearest_k = 8;
    std::vector<int> or_opt_segment_lengths = {1, 2, 3};
};

struct InterImproveOptions {
    std::vector<std::string> methods = {"2opt*", "insert", "swap", "cross"};
    int max_passes = 2;
    int lam = 3;
    int max_iterations = 30;
    int lambda_max_route_size = 20;
    int lambda_max_pairs_per_routes = 2500;
};

struct TabuOptions {
    int max_iterations = 1000;
    int tabu_tenure = 35;
    int neighborhood_samples = 50;
    int seed = 42;
    std::vector<std::string> moves = {
        "intra_2opt",
        "intra_relocate",
        "intra_swap",
        "inter_insert",
        "inter_swap",
        "two_opt_star",
        "cross",
        "lambda_interchange",
    };
    int lam = 3;
    bool aspiration = true;
    bool diversification = true;
    int diversification_limit = 400;
    int kick_strength = 4;
};

struct SimulatedAnnealingOptions {
    std::optional<double> initial_temperature = std::nullopt;
    double final_temperature = 1e-3;
    double cooling_rate = 0.995;
    int iterations_per_temperature = 100;
    int max_iterations = 5000;
    int seed = 42;
    std::vector<std::string> moves = {
        "intra_2opt",
        "intra_relocate",
        "intra_swap",
        "inter_insert",
        "inter_swap",
        "two_opt_star",
        "cross",
        "lambda_interchange",
    };
    int lam = 3;
    int max_neighbor_attempts = 50;
    bool adaptive_moves = true;
    bool reheat = false;
    int stagnation_limit = 2500;
    double reheat_factor = 1.5;
};

struct IteratedLocalSearchOptions {
    int max_iterations = 100;
    int perturbation_strength = 3;
    int local_search_rounds = 1;
    int seed = 42;
    std::string acceptance = "better";  // better | metropolis
    double acceptance_temperature = 1.0;
    double cooling_rate = 0.999;
    int lam = 3;
    std::vector<std::string> intra_methods = {"2opt", "or_opt", "relocate", "exchange"};
    std::vector<std::string> inter_methods = {"insert", "swap", "cross"};
    std::vector<std::string> perturbation_moves = {
        "intra_2opt", "intra_relocate", "inter_insert", "inter_swap", "two_opt_star"
    };
    bool diversification = true;
    int stagnation_limit = 80;
    int max_perturbation_strength = 10;
};

struct LargeNeighborhoodSearchOptions {
    int max_iterations = 100;
    double min_destroy_fraction = 0.10;
    double max_destroy_fraction = 0.30;
    int seed = 42;
    std::string acceptance = "metropolis";  // better | metropolis
    double acceptance_temperature = 10.0;
    double cooling_rate = 0.998;
    int lam = 3;
    bool use_local_search = true;
    int local_search_every = 3;
    bool diversification = true;
    int stagnation_limit = 60;
};

struct GeneticOptions {
    int population_size = 30;
    int generations = 50;
    double crossover_rate = 0.9;
    double mutation_rate = 0.25;
    int tournament_size = 3;
    int elitism = 2;
    int seed = 42;
    bool use_local_search = false;
    double local_search_prob = 0.15;
    int lam = 3;
    std::vector<std::string> intra_methods = {"2opt", "or_opt", "relocate", "exchange"};
    std::vector<std::string> inter_methods = {"insert", "swap", "cross"};
    bool diversification = true;
    int stagnation_limit = 35;
    double restart_fraction = 0.30;
};

struct MemeticOptions {
    int population_size = 30;
    int generations = 50;
    double crossover_rate = 0.9;
    double mutation_rate = 0.20;
    int tournament_size = 3;
    int elitism = 2;
    int seed = 42;
    int lam = 3;
    std::vector<std::string> intra_methods = {"2opt", "or_opt", "relocate", "exchange"};
    std::vector<std::string> inter_methods = {"insert", "swap", "cross"};
    double offspring_local_search_prob = 0.15;
    int elite_local_search_count = 2;
    int elite_local_search_period = 5;
    int intra_passes = 1;
    int inter_passes = 1;
    bool diversification = true;
    int stagnation_limit = 30;
    double restart_fraction = 0.25;
};

struct AcoOptions {
    int num_ants = 15;
    int iterations = 50;
    double alpha = 1.0;
    double beta = 3.0;
    double evaporation_rate = 0.15;
    double q0 = 0.20;
    double pheromone_init = 1.0;
    double elite_weight = 2.0;
    int seed = 42;
    bool use_local_search = false;
    double local_search_prob = 0.15;
    int lam = 3;
    double pheromone_min = 1e-4;
    double pheromone_max = 1e3;
};

Instance load_instance_file(const std::string& path);
Routes load_routes_file(const std::string& path);

void ensure_valid_instance(const Instance& instance);

double route_cost(const Route& route, const Instance& instance);
double solution_cost(const Routes& routes, const Instance& instance);
double route_load(const Route& route, const Instance& instance);

bool feasible_solution(const Routes& routes, const Instance& instance);

Routes nearest_neighbor(const Instance& instance);
Routes clarke_wright(const Instance& instance);
Routes cheapest_insertion(const Instance& instance);
Routes sweep(const Instance& instance);

Routes improve_intra(const Instance& instance, const Routes& routes, const IntraImproveOptions& options = {});
Routes improve_inter(const Instance& instance, const Routes& routes, const InterImproveOptions& options = {});

SolverOutput tabu_search(
    const Instance& instance,
    const std::optional<Routes>& initial_routes = std::nullopt,
    const TabuOptions& options = {}
);

SolverOutput simulated_annealing(
    const Instance& instance,
    const std::optional<Routes>& initial_routes = std::nullopt,
    const SimulatedAnnealingOptions& options = {}
);

SolverOutput iterated_local_search(
    const Instance& instance,
    const std::optional<Routes>& initial_routes = std::nullopt,
    const IteratedLocalSearchOptions& options = {}
);

SolverOutput large_neighborhood_search(
    const Instance& instance,
    const std::optional<Routes>& initial_routes = std::nullopt,
    const LargeNeighborhoodSearchOptions& options = {}
);

SolverOutput genetic_algorithm(
    const Instance& instance,
    const std::optional<Routes>& initial_routes = std::nullopt,
    const GeneticOptions& options = {}
);

SolverOutput memetic_algorithm(
    const Instance& instance,
    const std::optional<Routes>& initial_routes = std::nullopt,
    const MemeticOptions& options = {}
);

SolverOutput ant_colony_optimization(
    const Instance& instance,
    const std::optional<Routes>& initial_routes = std::nullopt,
    const AcoOptions& options = {}
);

SolverOutput output_from_routes(const Instance& instance, const Routes& routes);
std::string routes_to_json(const Routes& routes);
std::string solver_output_to_json(const SolverOutput& output);
std::string error_to_json(const std::string& message);

std::unordered_map<std::string, std::string> parse_cli_args(int argc, char** argv);
std::vector<std::string> split_csv(const std::string& csv);
bool as_bool(const std::string& value);

}  // namespace cvrp

