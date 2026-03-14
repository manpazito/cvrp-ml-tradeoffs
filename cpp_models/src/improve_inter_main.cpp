#include "cvrp_core.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace cvrp;

    try {
        const auto args = parse_cli_args(argc, argv);
        if (!args.count("instance")) {
            throw std::runtime_error("Missing required --instance <path>");
        }

        const Instance instance = load_instance_file(args.at("instance"));
        Routes routes = args.count("routes") ? load_routes_file(args.at("routes")) : nearest_neighbor(instance);

        InterImproveOptions options;
        if (args.count("methods")) {
            options.methods = split_csv(args.at("methods"));
        }
        if (args.count("max-passes")) {
            options.max_passes = std::stoi(args.at("max-passes"));
        }
        if (args.count("max-iterations")) {
            options.max_iterations = std::stoi(args.at("max-iterations"));
        }
        if (args.count("lam")) {
            options.lam = std::stoi(args.at("lam"));
        }
        if (args.count("lambda-max-route-size")) {
            options.lambda_max_route_size = std::stoi(args.at("lambda-max-route-size"));
        }
        if (args.count("lambda-max-pairs")) {
            options.lambda_max_pairs_per_routes = std::stoi(args.at("lambda-max-pairs"));
        }

        const Routes improved = improve_inter(instance, routes, options);
        std::cout << solver_output_to_json(output_from_routes(instance, improved)) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

