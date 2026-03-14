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

        SimulatedAnnealingOptions options;
        if (args.count("initial-temperature")) {
            options.initial_temperature = std::stod(args.at("initial-temperature"));
        }
        if (args.count("final-temperature")) {
            options.final_temperature = std::stod(args.at("final-temperature"));
        }
        if (args.count("cooling-rate")) {
            options.cooling_rate = std::stod(args.at("cooling-rate"));
        }
        if (args.count("iterations-per-temperature")) {
            options.iterations_per_temperature = std::stoi(args.at("iterations-per-temperature"));
        }
        if (args.count("max-iterations")) {
            options.max_iterations = std::stoi(args.at("max-iterations"));
        }
        if (args.count("seed")) {
            options.seed = std::stoi(args.at("seed"));
        }
        if (args.count("moves")) {
            options.moves = split_csv(args.at("moves"));
        }
        if (args.count("lam")) {
            options.lam = std::stoi(args.at("lam"));
        }
        if (args.count("max-neighbor-attempts")) {
            options.max_neighbor_attempts = std::stoi(args.at("max-neighbor-attempts"));
        }
        if (args.count("adaptive-moves")) {
            options.adaptive_moves = as_bool(args.at("adaptive-moves"));
        }
        if (args.count("reheat")) {
            options.reheat = as_bool(args.at("reheat"));
        }
        if (args.count("stagnation-limit")) {
            options.stagnation_limit = std::stoi(args.at("stagnation-limit"));
        }
        if (args.count("reheat-factor")) {
            options.reheat_factor = std::stod(args.at("reheat-factor"));
        }

        const SolverOutput output = simulated_annealing(instance, initial_routes, options);
        std::cout << solver_output_to_json(output) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

