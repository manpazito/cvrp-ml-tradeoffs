#include "cvrp_core.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace cvrp;

    try {
        const auto args = parse_cli_args(argc, argv);
        if (!args.count("instance")) {
            throw std::runtime_error("Missing required --instance <path>");
        }

        const Instance instance = load_instance_file(args.at("instance"));
        std::optional<Routes> initial_routes = std::nullopt;
        if (args.count("routes")) {
            initial_routes = load_routes_file(args.at("routes"));
        }

        AcoOptions options;
        if (args.count("num-ants")) {
            options.num_ants = std::stoi(args.at("num-ants"));
        }
        if (args.count("iterations")) {
            options.iterations = std::stoi(args.at("iterations"));
        }
        if (args.count("alpha")) {
            options.alpha = std::stod(args.at("alpha"));
        }
        if (args.count("beta")) {
            options.beta = std::stod(args.at("beta"));
        }
        if (args.count("evaporation-rate")) {
            options.evaporation_rate = std::stod(args.at("evaporation-rate"));
        }
        if (args.count("q0")) {
            options.q0 = std::stod(args.at("q0"));
        }
        if (args.count("pheromone-init")) {
            options.pheromone_init = std::stod(args.at("pheromone-init"));
        }
        if (args.count("elite-weight")) {
            options.elite_weight = std::stod(args.at("elite-weight"));
        }
        if (args.count("seed")) {
            options.seed = std::stoi(args.at("seed"));
        }
        if (args.count("use-local-search")) {
            options.use_local_search = as_bool(args.at("use-local-search"));
        }
        if (args.count("local-search-prob")) {
            options.local_search_prob = std::stod(args.at("local-search-prob"));
        }
        if (args.count("lam")) {
            options.lam = std::stoi(args.at("lam"));
        }
        if (args.count("pheromone-min")) {
            options.pheromone_min = std::stod(args.at("pheromone-min"));
        }
        if (args.count("pheromone-max")) {
            options.pheromone_max = std::stod(args.at("pheromone-max"));
        }

        const SolverOutput output = ant_colony_optimization(instance, initial_routes, options);
        std::cout << solver_output_to_json(output) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

