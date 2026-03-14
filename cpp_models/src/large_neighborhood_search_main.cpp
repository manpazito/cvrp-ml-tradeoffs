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

        LargeNeighborhoodSearchOptions options;
        if (args.count("max-iterations")) {
            options.max_iterations = std::stoi(args.at("max-iterations"));
        }
        if (args.count("min-destroy-fraction")) {
            options.min_destroy_fraction = std::stod(args.at("min-destroy-fraction"));
        }
        if (args.count("max-destroy-fraction")) {
            options.max_destroy_fraction = std::stod(args.at("max-destroy-fraction"));
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
        if (args.count("use-local-search")) {
            options.use_local_search = as_bool(args.at("use-local-search"));
        }
        if (args.count("local-search-every")) {
            options.local_search_every = std::stoi(args.at("local-search-every"));
        }
        if (args.count("diversification")) {
            options.diversification = as_bool(args.at("diversification"));
        }
        if (args.count("stagnation-limit")) {
            options.stagnation_limit = std::stoi(args.at("stagnation-limit"));
        }

        const SolverOutput output = large_neighborhood_search(instance, initial_routes, options);
        std::cout << solver_output_to_json(output) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

