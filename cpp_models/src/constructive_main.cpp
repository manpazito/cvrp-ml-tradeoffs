#include "cvrp_core.hpp"

#include <iostream>
#include <stdexcept>

namespace {
std::string normalize_method(std::string method) {
    for (char& c : method) {
        if (c == '-') {
            c = '_';
        }
    }
    return method;
}
}

int main(int argc, char** argv) {
    using namespace cvrp;

    try {
        const auto args = parse_cli_args(argc, argv);
        if (!args.count("instance")) {
            throw std::runtime_error("Missing required --instance <path>");
        }

        const Instance instance = load_instance_file(args.at("instance"));
        const std::string method = normalize_method(args.count("method") ? args.at("method") : "nearest_neighbor");

        Routes routes;
        if (method == "nearest_neighbor") {
            routes = nearest_neighbor(instance);
        } else if (method == "clarke_wright") {
            routes = clarke_wright(instance);
        } else if (method == "cheapest_insertion") {
            routes = cheapest_insertion(instance);
        } else if (method == "sweep") {
            routes = sweep(instance);
        } else {
            throw std::runtime_error("Unknown constructive method: " + method);
        }

        std::cout << solver_output_to_json(output_from_routes(instance, routes)) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << cvrp::error_to_json(ex.what()) << '\n';
        return 1;
    }
}

