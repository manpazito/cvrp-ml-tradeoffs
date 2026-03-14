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

        MemeticOptions options;
        if (args.count("population-size")) {
            options.population_size = std::stoi(args.at("population-size"));
        }
        if (args.count("generations")) {
            options.generations = std::stoi(args.at("generations"));
        }
        if (args.count("crossover-rate")) {
            options.crossover_rate = std::stod(args.at("crossover-rate"));
        }
        if (args.count("mutation-rate")) {
            options.mutation_rate = std::stod(args.at("mutation-rate"));
        }
        if (args.count("tournament-size")) {
            options.tournament_size = std::stoi(args.at("tournament-size"));
        }
        if (args.count("elitism")) {
            options.elitism = std::stoi(args.at("elitism"));
        }
        if (args.count("seed")) {
            options.seed = std::stoi(args.at("seed"));
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
        if (args.count("offspring-local-search-prob")) {
            options.offspring_local_search_prob = std::stod(args.at("offspring-local-search-prob"));
        }
        if (args.count("elite-local-search-count")) {
            options.elite_local_search_count = std::stoi(args.at("elite-local-search-count"));
        }
        if (args.count("elite-local-search-period")) {
            options.elite_local_search_period = std::stoi(args.at("elite-local-search-period"));
        }
        if (args.count("intra-passes")) {
            options.intra_passes = std::stoi(args.at("intra-passes"));
        }
        if (args.count("inter-passes")) {
            options.inter_passes = std::stoi(args.at("inter-passes"));
        }
        if (args.count("diversification")) {
            options.diversification = as_bool(args.at("diversification"));
        }
        if (args.count("stagnation-limit")) {
            options.stagnation_limit = std::stoi(args.at("stagnation-limit"));
        }
        if (args.count("restart-fraction")) {
            options.restart_fraction = std::stod(args.at("restart-fraction"));
        }

        const SolverOutput output = memetic_algorithm(instance, initial_routes, options);
        std::cout << solver_output_to_json(output) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

