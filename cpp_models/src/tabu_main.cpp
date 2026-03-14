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

        TabuOptions options;
        if (args.count("max-iterations")) {
            options.max_iterations = std::stoi(args.at("max-iterations"));
        }
        if (args.count("tabu-tenure")) {
            options.tabu_tenure = std::stoi(args.at("tabu-tenure"));
        }
        if (args.count("neighborhood-samples")) {
            options.neighborhood_samples = std::stoi(args.at("neighborhood-samples"));
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
        if (args.count("aspiration")) {
            options.aspiration = as_bool(args.at("aspiration"));
        }
        if (args.count("diversification")) {
            options.diversification = as_bool(args.at("diversification"));
        }
        if (args.count("diversification-limit")) {
            options.diversification_limit = std::stoi(args.at("diversification-limit"));
        }
        if (args.count("kick-strength")) {
            options.kick_strength = std::stoi(args.at("kick-strength"));
        }

        const SolverOutput output = tabu_search(instance, initial_routes, options);
        std::cout << solver_output_to_json(output) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

