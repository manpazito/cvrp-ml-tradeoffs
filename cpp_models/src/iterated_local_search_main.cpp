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

        IteratedLocalSearchOptions options;
        if (args.count("max-iterations")) {
            options.max_iterations = std::stoi(args.at("max-iterations"));
        }
        if (args.count("perturbation-strength")) {
            options.perturbation_strength = std::stoi(args.at("perturbation-strength"));
        }
        if (args.count("local-search-rounds")) {
            options.local_search_rounds = std::stoi(args.at("local-search-rounds"));
        }
        if (args.count("seed")) {
            options.seed = std::stoi(args.at("seed"));
        }
        if (args.count("acceptance")) {
            options.acceptance = args.at("acceptance");
        }
        if (args.count("acceptance-temperature")) {
            options.acceptance_temperature = std::stod(args.at("acceptance-temperature"));
        }
        if (args.count("cooling-rate")) {
            options.cooling_rate = std::stod(args.at("cooling-rate"));
        }
        if (args.count("lam")) {
            options.lam = std::stoi(args.at("lam"));
        }
        if (args.count("intra-methods")) {
            options.intra_methods = split_csv(args.at("intra-methods"));
        }
        if (args.count("inter-methods")) {
            options.inter_methods = split_csv(args.at("inter-methods"));
        }
        if (args.count("perturbation-moves")) {
            options.perturbation_moves = split_csv(args.at("perturbation-moves"));
        }
        if (args.count("diversification")) {
            options.diversification = as_bool(args.at("diversification"));
        }
        if (args.count("stagnation-limit")) {
            options.stagnation_limit = std::stoi(args.at("stagnation-limit"));
        }
        if (args.count("max-perturbation-strength")) {
            options.max_perturbation_strength = std::stoi(args.at("max-perturbation-strength"));
        }

        const SolverOutput output = iterated_local_search(instance, initial_routes, options);
        std::cout << solver_output_to_json(output) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

