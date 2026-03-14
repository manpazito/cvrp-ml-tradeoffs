#include "cvrp_core.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace cvrp {
namespace {

constexpr double kEps = 1e-9;

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string trim(const std::string& input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

bool starts_with(const std::string& value, const std::string& prefix) {
    if (value.size() < prefix.size()) {
        return false;
    }
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool try_parse_int(const std::string& token, int* out) {
    if (token.empty()) {
        return false;
    }
    char* end = nullptr;
    const long value = std::strtol(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

bool try_parse_double(const std::string& token, double* out) {
    if (token.empty()) {
        return false;
    }
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

bool read_next_token(std::istream& in, std::string* token) {
    while (*token = "", in >> *token) {
        if (!token->empty() && (*token)[0] == '#') {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        return true;
    }
    return false;
}

double euclidean(const Point& a, const Point& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

Point customer_point(const Instance& instance, CustomerId id) {
    auto it = instance.coords.find(id);
    if (it == instance.coords.end()) {
        throw std::runtime_error("Unknown customer id in route: " + std::to_string(id));
    }
    return it->second;
}

double customer_demand(const Instance& instance, CustomerId id) {
    auto it = instance.demand.find(id);
    if (it == instance.demand.end()) {
        throw std::runtime_error("Unknown customer id in route demand lookup: " + std::to_string(id));
    }
    return it->second;
}

std::vector<CustomerId> flatten_routes(const Routes& routes) {
    std::vector<CustomerId> out;
    for (const auto& route : routes) {
        out.insert(out.end(), route.begin(), route.end());
    }
    return out;
}

Routes cleanup_routes(const Routes& routes) {
    Routes out;
    out.reserve(routes.size());
    for (const auto& route : routes) {
        if (!route.empty()) {
            out.push_back(route);
        }
    }
    return out;
}

bool is_feasible_partial_route(const Route& route, const Instance& instance) {
    return route_load(route, instance) <= instance.capacity + kEps;
}

int rand_int(std::mt19937& rng, int low, int high_inclusive) {
    if (low > high_inclusive) {
        throw std::runtime_error("Invalid random bounds");
    }
    std::uniform_int_distribution<int> dist(low, high_inclusive);
    return dist(rng);
}

double rand01(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

template <typename T>
const T& random_choice(const std::vector<T>& values, std::mt19937& rng) {
    if (values.empty()) {
        throw std::runtime_error("random_choice called on empty container");
    }
    return values[static_cast<std::size_t>(rand_int(rng, 0, static_cast<int>(values.size()) - 1))];
}

std::string route_key(const Route& route) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < route.size(); ++i) {
        if (i) {
            oss << ',';
        }
        oss << route[i];
    }
    return oss.str();
}

std::string routes_signature(const Routes& routes) {
    std::vector<Route> normalized = routes;
    std::sort(normalized.begin(), normalized.end());
    std::ostringstream oss;
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        if (i) {
            oss << '|';
        }
        oss << route_key(normalized[i]);
    }
    return oss.str();
}

int find_route_with_customer(const Routes& routes, CustomerId cid) {
    for (std::size_t i = 0; i < routes.size(); ++i) {
        const auto& route = routes[i];
        if (std::find(route.begin(), route.end(), cid) != route.end()) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::string permutation_key(const std::vector<CustomerId>& perm) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (i) {
            oss << ',';
        }
        oss << perm[i];
    }
    return oss.str();
}

Route reverse_slice(const Route& route, int begin_inclusive, int end_inclusive) {
    Route out = route;
    std::reverse(out.begin() + begin_inclusive, out.begin() + end_inclusive + 1);
    return out;
}

std::vector<std::vector<int>> subset_index_tuples(int n, int lam) {
    std::vector<std::vector<int>> out;
    out.push_back({});

    const int max_size = std::min(n, lam);
    std::vector<int> current;

    std::function<void(int, int, int)> rec = [&](int start, int k, int target) {
        if (k == target) {
            out.push_back(current);
            return;
        }
        for (int i = start; i < n; ++i) {
            current.push_back(i);
            rec(i + 1, k + 1, target);
            current.pop_back();
        }
    };

    for (int size = 1; size <= max_size; ++size) {
        rec(0, 0, size);
    }

    return out;
}

void remove_by_indices(const Route& route, const std::vector<int>& indices, Route* picked, Route* remain) {
    std::unordered_set<int> idx_set(indices.begin(), indices.end());
    picked->clear();
    remain->clear();
    picked->reserve(indices.size());
    remain->reserve(route.size() - indices.size());

    for (int i = 0; i < static_cast<int>(route.size()); ++i) {
        if (idx_set.count(i)) {
            picked->push_back(route[static_cast<std::size_t>(i)]);
        } else {
            remain->push_back(route[static_cast<std::size_t>(i)]);
        }
    }
}

std::vector<int> sample_indices(int n, int k, std::mt19937& rng) {
    std::vector<int> values(n);
    std::iota(values.begin(), values.end(), 0);
    std::shuffle(values.begin(), values.end(), rng);
    values.resize(static_cast<std::size_t>(k));
    std::sort(values.begin(), values.end());
    return values;
}

int pick_route_index(const Routes& routes, std::mt19937& rng, int min_size = 1, int different_from = -1) {
    std::vector<int> valid;
    for (int i = 0; i < static_cast<int>(routes.size()); ++i) {
        if (i == different_from) {
            continue;
        }
        if (static_cast<int>(routes[static_cast<std::size_t>(i)].size()) >= min_size) {
            valid.push_back(i);
        }
    }
    if (valid.empty()) {
        return -1;
    }
    return random_choice(valid, rng);
}

// -------------------- Intra-route moves --------------------

std::pair<double, Route> best_two_opt_move(const Route& route, const Instance& instance) {
    const int n = static_cast<int>(route.size());
    if (n < 4) {
        return {0.0, route};
    }

    const double base = route_cost(route, instance);
    double best_delta = 0.0;
    Route best_route = route;

    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 2; j < n; ++j) {
            if (i == 0 && j == n - 1) {
                continue;
            }
            Route candidate = route;
            std::reverse(candidate.begin() + i + 1, candidate.begin() + j + 1);
            const double cand = route_cost(candidate, instance);
            const double delta = cand - base;
            if (delta < best_delta) {
                best_delta = delta;
                best_route = std::move(candidate);
            }
        }
    }

    return {best_delta, best_route};
}

std::pair<double, Route> best_three_opt_move(const Route& route, const Instance& instance) {
    const int n = static_cast<int>(route.size());
    if (n < 6) {
        return {0.0, route};
    }

    const double base = route_cost(route, instance);
    double best_delta = 0.0;
    Route best_route = route;

    for (int i = 0; i < n - 5; ++i) {
        for (int j = i + 2; j < n - 3; ++j) {
            for (int k = j + 2; k < n - 1; ++k) {
                Route a(route.begin(), route.begin() + i + 1);
                Route b(route.begin() + i + 1, route.begin() + j + 1);
                Route c(route.begin() + j + 1, route.begin() + k + 1);
                Route d(route.begin() + k + 1, route.end());

                auto make_candidate = [&](const Route& b_part, const Route& c_part) {
                    Route cand;
                    cand.reserve(route.size());
                    cand.insert(cand.end(), a.begin(), a.end());
                    cand.insert(cand.end(), b_part.begin(), b_part.end());
                    cand.insert(cand.end(), c_part.begin(), c_part.end());
                    cand.insert(cand.end(), d.begin(), d.end());
                    return cand;
                };

                Route b_rev = b;
                Route c_rev = c;
                std::reverse(b_rev.begin(), b_rev.end());
                std::reverse(c_rev.begin(), c_rev.end());

                std::vector<Route> candidates;
                candidates.push_back(make_candidate(b_rev, c));
                candidates.push_back(make_candidate(b, c_rev));
                candidates.push_back(make_candidate(b_rev, c_rev));

                {
                    Route cand;
                    cand.reserve(route.size());
                    cand.insert(cand.end(), a.begin(), a.end());
                    cand.insert(cand.end(), c.begin(), c.end());
                    cand.insert(cand.end(), b.begin(), b.end());
                    cand.insert(cand.end(), d.begin(), d.end());
                    candidates.push_back(std::move(cand));
                }
                {
                    Route cand;
                    cand.reserve(route.size());
                    cand.insert(cand.end(), a.begin(), a.end());
                    cand.insert(cand.end(), c_rev.begin(), c_rev.end());
                    cand.insert(cand.end(), b.begin(), b.end());
                    cand.insert(cand.end(), d.begin(), d.end());
                    candidates.push_back(std::move(cand));
                }
                {
                    Route cand;
                    cand.reserve(route.size());
                    cand.insert(cand.end(), a.begin(), a.end());
                    cand.insert(cand.end(), c.begin(), c.end());
                    cand.insert(cand.end(), b_rev.begin(), b_rev.end());
                    cand.insert(cand.end(), d.begin(), d.end());
                    candidates.push_back(std::move(cand));
                }
                {
                    Route cand;
                    cand.reserve(route.size());
                    cand.insert(cand.end(), a.begin(), a.end());
                    cand.insert(cand.end(), c_rev.begin(), c_rev.end());
                    cand.insert(cand.end(), b_rev.begin(), b_rev.end());
                    cand.insert(cand.end(), d.begin(), d.end());
                    candidates.push_back(std::move(cand));
                }

                for (const auto& candidate : candidates) {
                    const double cand = route_cost(candidate, instance);
                    const double delta = cand - base;
                    if (delta < best_delta) {
                        best_delta = delta;
                        best_route = candidate;
                    }
                }
            }
        }
    }

    return {best_delta, best_route};
}

std::pair<double, Route> best_or_opt_move(
    const Route& route,
    const Instance& instance,
    const std::vector<int>& segment_lengths
) {
    const int n = static_cast<int>(route.size());
    if (n < 3) {
        return {0.0, route};
    }

    const double base = route_cost(route, instance);
    double best_delta = 0.0;
    Route best_route = route;

    for (int seg_len : segment_lengths) {
        if (seg_len <= 0 || seg_len >= n) {
            continue;
        }

        for (int i = 0; i <= n - seg_len; ++i) {
            Route segment(route.begin() + i, route.begin() + i + seg_len);
            Route remainder;
            remainder.reserve(route.size() - seg_len);
            remainder.insert(remainder.end(), route.begin(), route.begin() + i);
            remainder.insert(remainder.end(), route.begin() + i + seg_len, route.end());

            for (int j = 0; j <= static_cast<int>(remainder.size()); ++j) {
                if (j == i) {
                    continue;
                }
                Route candidate;
                candidate.reserve(route.size());
                candidate.insert(candidate.end(), remainder.begin(), remainder.begin() + j);
                candidate.insert(candidate.end(), segment.begin(), segment.end());
                candidate.insert(candidate.end(), remainder.begin() + j, remainder.end());

                const double cand = route_cost(candidate, instance);
                const double delta = cand - base;
                if (delta < best_delta) {
                    best_delta = delta;
                    best_route = std::move(candidate);
                }
            }
        }
    }

    return {best_delta, best_route};
}

std::pair<double, Route> best_relocate_move(const Route& route, const Instance& instance) {
    static const std::vector<int> segment_lengths = {1};
    return best_or_opt_move(route, instance, segment_lengths);
}

std::pair<double, Route> best_exchange_move(const Route& route, const Instance& instance) {
    const int n = static_cast<int>(route.size());
    if (n < 2) {
        return {0.0, route};
    }

    const double base = route_cost(route, instance);
    double best_delta = 0.0;
    Route best_route = route;

    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Route candidate = route;
            std::swap(candidate[static_cast<std::size_t>(i)], candidate[static_cast<std::size_t>(j)]);
            const double cand = route_cost(candidate, instance);
            const double delta = cand - base;
            if (delta < best_delta) {
                best_delta = delta;
                best_route = std::move(candidate);
            }
        }
    }

    return {best_delta, best_route};
}

std::unordered_set<CustomerId> nearest_nodes(
    CustomerId node,
    const Route& pool,
    const Instance& instance,
    int k
) {
    if (k <= 0) {
        return {};
    }

    std::vector<std::pair<double, CustomerId>> scored;
    scored.reserve(pool.size());
    const Point node_point = customer_point(instance, node);

    for (CustomerId other : pool) {
        if (other == node) {
            continue;
        }
        scored.emplace_back(euclidean(node_point, customer_point(instance, other)), other);
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::unordered_set<CustomerId> out;
    const int take = std::min(k, static_cast<int>(scored.size()));
    for (int i = 0; i < take; ++i) {
        out.insert(scored[static_cast<std::size_t>(i)].second);
    }
    return out;
}

std::pair<double, Route> best_geni_move(const Route& route, const Instance& instance, int nearest_k) {
    const int n = static_cast<int>(route.size());
    if (n < 3) {
        return {0.0, route};
    }

    const double base = route_cost(route, instance);
    double best_delta = 0.0;
    Route best_route = route;

    for (int i = 0; i < n; ++i) {
        const CustomerId node = route[static_cast<std::size_t>(i)];
        Route reduced;
        reduced.reserve(route.size() - 1);
        reduced.insert(reduced.end(), route.begin(), route.begin() + i);
        reduced.insert(reduced.end(), route.begin() + i + 1, route.end());

        const auto nearest = nearest_nodes(node, reduced, instance, nearest_k);

        for (int j = 0; j <= static_cast<int>(reduced.size()); ++j) {
            CustomerId prev_node = (j > 0) ? reduced[static_cast<std::size_t>(j - 1)] : -1;
            CustomerId next_node = (j < static_cast<int>(reduced.size())) ? reduced[static_cast<std::size_t>(j)] : -1;

            if (!nearest.empty()) {
                const bool prev_ok = (prev_node != -1 && nearest.count(prev_node));
                const bool next_ok = (next_node != -1 && nearest.count(next_node));
                if (!prev_ok && !next_ok) {
                    continue;
                }
            }

            Route candidate;
            candidate.reserve(route.size());
            candidate.insert(candidate.end(), reduced.begin(), reduced.begin() + j);
            candidate.push_back(node);
            candidate.insert(candidate.end(), reduced.begin() + j, reduced.end());

            const double cand = route_cost(candidate, instance);
            const double delta = cand - base;
            if (delta < best_delta) {
                best_delta = delta;
                best_route = std::move(candidate);
            }
        }
    }

    return {best_delta, best_route};
}

Route iterate_route_improvement(
    const Route& route,
    const std::function<std::pair<double, Route>(const Route&)>& move_fn,
    int max_iterations
) {
    Route current = route;
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto [delta, candidate] = move_fn(current);
        if (delta < -kEps) {
            current = std::move(candidate);
        } else {
            break;
        }
    }
    return current;
}

// -------------------- Inter-route moves --------------------

std::pair<double, Routes> best_two_opt_star_move(const Routes& routes, const Instance& instance) {
    double best_delta = 0.0;
    Routes best_routes = routes;

    for (int ra = 0; ra < static_cast<int>(routes.size()) - 1; ++ra) {
        for (int rb = ra + 1; rb < static_cast<int>(routes.size()); ++rb) {
            const auto& route_a = routes[static_cast<std::size_t>(ra)];
            const auto& route_b = routes[static_cast<std::size_t>(rb)];
            const double base_cost = route_cost(route_a, instance) + route_cost(route_b, instance);

            for (int cut_a = 0; cut_a <= static_cast<int>(route_a.size()); ++cut_a) {
                Route head_a(route_a.begin(), route_a.begin() + cut_a);
                Route tail_a(route_a.begin() + cut_a, route_a.end());

                for (int cut_b = 0; cut_b <= static_cast<int>(route_b.size()); ++cut_b) {
                    Route head_b(route_b.begin(), route_b.begin() + cut_b);
                    Route tail_b(route_b.begin() + cut_b, route_b.end());

                    Route cand_a = head_a;
                    cand_a.insert(cand_a.end(), tail_b.begin(), tail_b.end());

                    Route cand_b = head_b;
                    cand_b.insert(cand_b.end(), tail_a.begin(), tail_a.end());

                    if (!is_feasible_partial_route(cand_a, instance)) {
                        continue;
                    }
                    if (!is_feasible_partial_route(cand_b, instance)) {
                        continue;
                    }

                    const double cand_cost = route_cost(cand_a, instance) + route_cost(cand_b, instance);
                    const double delta = cand_cost - base_cost;
                    if (delta < best_delta) {
                        Routes candidate = routes;
                        candidate[static_cast<std::size_t>(ra)] = std::move(cand_a);
                        candidate[static_cast<std::size_t>(rb)] = std::move(cand_b);
                        best_delta = delta;
                        best_routes = cleanup_routes(candidate);
                    }
                }
            }
        }
    }

    return {best_delta, best_routes};
}

std::pair<double, Routes> best_insert_move(const Routes& routes, const Instance& instance) {
    double best_delta = 0.0;
    Routes best_routes = routes;

    for (int ra = 0; ra < static_cast<int>(routes.size()); ++ra) {
        const auto& route_a = routes[static_cast<std::size_t>(ra)];
        if (route_a.empty()) {
            continue;
        }

        for (int rb = 0; rb < static_cast<int>(routes.size()); ++rb) {
            if (ra == rb) {
                continue;
            }
            const auto& route_b = routes[static_cast<std::size_t>(rb)];
            const double base_cost = route_cost(route_a, instance) + route_cost(route_b, instance);

            for (int i = 0; i < static_cast<int>(route_a.size()); ++i) {
                const CustomerId node = route_a[static_cast<std::size_t>(i)];
                Route cand_a;
                cand_a.reserve(route_a.size() - 1);
                cand_a.insert(cand_a.end(), route_a.begin(), route_a.begin() + i);
                cand_a.insert(cand_a.end(), route_a.begin() + i + 1, route_a.end());

                for (int j = 0; j <= static_cast<int>(route_b.size()); ++j) {
                    Route cand_b = route_b;
                    cand_b.insert(cand_b.begin() + j, node);

                    if (!is_feasible_partial_route(cand_a, instance)) {
                        continue;
                    }
                    if (!is_feasible_partial_route(cand_b, instance)) {
                        continue;
                    }

                    const double cand_cost = route_cost(cand_a, instance) + route_cost(cand_b, instance);
                    const double delta = cand_cost - base_cost;
                    if (delta < best_delta) {
                        Routes candidate = routes;
                        candidate[static_cast<std::size_t>(ra)] = std::move(cand_a);
                        candidate[static_cast<std::size_t>(rb)] = std::move(cand_b);
                        best_delta = delta;
                        best_routes = cleanup_routes(candidate);
                    }
                }
            }
        }
    }

    return {best_delta, best_routes};
}

std::pair<double, Routes> best_swap_move(const Routes& routes, const Instance& instance) {
    double best_delta = 0.0;
    Routes best_routes = routes;

    for (int ra = 0; ra < static_cast<int>(routes.size()) - 1; ++ra) {
        for (int rb = ra + 1; rb < static_cast<int>(routes.size()); ++rb) {
            const auto& route_a = routes[static_cast<std::size_t>(ra)];
            const auto& route_b = routes[static_cast<std::size_t>(rb)];
            if (route_a.empty() || route_b.empty()) {
                continue;
            }

            const double base_cost = route_cost(route_a, instance) + route_cost(route_b, instance);
            const double base_load_a = route_load(route_a, instance);
            const double base_load_b = route_load(route_b, instance);

            for (int i = 0; i < static_cast<int>(route_a.size()); ++i) {
                for (int j = 0; j < static_cast<int>(route_b.size()); ++j) {
                    const CustomerId node_a = route_a[static_cast<std::size_t>(i)];
                    const CustomerId node_b = route_b[static_cast<std::size_t>(j)];

                    const double cand_load_a = base_load_a - customer_demand(instance, node_a) + customer_demand(instance, node_b);
                    const double cand_load_b = base_load_b - customer_demand(instance, node_b) + customer_demand(instance, node_a);
                    if (cand_load_a > instance.capacity + kEps || cand_load_b > instance.capacity + kEps) {
                        continue;
                    }

                    Route cand_a = route_a;
                    Route cand_b = route_b;
                    std::swap(cand_a[static_cast<std::size_t>(i)], cand_b[static_cast<std::size_t>(j)]);

                    const double cand_cost = route_cost(cand_a, instance) + route_cost(cand_b, instance);
                    const double delta = cand_cost - base_cost;
                    if (delta < best_delta) {
                        Routes candidate = routes;
                        candidate[static_cast<std::size_t>(ra)] = std::move(cand_a);
                        candidate[static_cast<std::size_t>(rb)] = std::move(cand_b);
                        best_delta = delta;
                        best_routes = cleanup_routes(candidate);
                    }
                }
            }
        }
    }

    return {best_delta, best_routes};
}

std::pair<double, Routes> best_cross_move(const Routes& routes, const Instance& instance, int lam) {
    double best_delta = 0.0;
    Routes best_routes = routes;

    for (int ra = 0; ra < static_cast<int>(routes.size()) - 1; ++ra) {
        for (int rb = ra + 1; rb < static_cast<int>(routes.size()); ++rb) {
            const auto& route_a = routes[static_cast<std::size_t>(ra)];
            const auto& route_b = routes[static_cast<std::size_t>(rb)];
            const double base_cost = route_cost(route_a, instance) + route_cost(route_b, instance);

            for (int i = 0; i <= static_cast<int>(route_a.size()); ++i) {
                for (int len_a = 0; len_a <= lam; ++len_a) {
                    if (i + len_a > static_cast<int>(route_a.size())) {
                        continue;
                    }

                    Route seg_a(route_a.begin() + i, route_a.begin() + i + len_a);
                    Route rem_a;
                    rem_a.reserve(route_a.size() - len_a);
                    rem_a.insert(rem_a.end(), route_a.begin(), route_a.begin() + i);
                    rem_a.insert(rem_a.end(), route_a.begin() + i + len_a, route_a.end());

                    for (int j = 0; j <= static_cast<int>(route_b.size()); ++j) {
                        for (int len_b = 0; len_b <= lam; ++len_b) {
                            if (j + len_b > static_cast<int>(route_b.size())) {
                                continue;
                            }
                            if (len_a == 0 && len_b == 0) {
                                continue;
                            }

                            Route seg_b(route_b.begin() + j, route_b.begin() + j + len_b);
                            Route rem_b;
                            rem_b.reserve(route_b.size() - len_b);
                            rem_b.insert(rem_b.end(), route_b.begin(), route_b.begin() + j);
                            rem_b.insert(rem_b.end(), route_b.begin() + j + len_b, route_b.end());

                            Route cand_a;
                            cand_a.reserve(rem_a.size() + seg_b.size());
                            cand_a.insert(cand_a.end(), rem_a.begin(), rem_a.begin() + i);
                            cand_a.insert(cand_a.end(), seg_b.begin(), seg_b.end());
                            cand_a.insert(cand_a.end(), rem_a.begin() + i, rem_a.end());

                            Route cand_b;
                            cand_b.reserve(rem_b.size() + seg_a.size());
                            cand_b.insert(cand_b.end(), rem_b.begin(), rem_b.begin() + j);
                            cand_b.insert(cand_b.end(), seg_a.begin(), seg_a.end());
                            cand_b.insert(cand_b.end(), rem_b.begin() + j, rem_b.end());

                            if (!is_feasible_partial_route(cand_a, instance)) {
                                continue;
                            }
                            if (!is_feasible_partial_route(cand_b, instance)) {
                                continue;
                            }

                            const double cand_cost = route_cost(cand_a, instance) + route_cost(cand_b, instance);
                            const double delta = cand_cost - base_cost;
                            if (delta < best_delta) {
                                Routes candidate = routes;
                                candidate[static_cast<std::size_t>(ra)] = std::move(cand_a);
                                candidate[static_cast<std::size_t>(rb)] = std::move(cand_b);
                                best_delta = delta;
                                best_routes = cleanup_routes(candidate);
                            }
                        }
                    }
                }
            }
        }
    }

    return {best_delta, best_routes};
}

std::pair<double, Routes> best_lambda_interchange_move(
    const Routes& routes,
    const Instance& instance,
    int lam,
    int max_route_size,
    int max_pairs_per_routes
) {
    double best_delta = 0.0;
    Routes best_routes = routes;

    for (int ra = 0; ra < static_cast<int>(routes.size()) - 1; ++ra) {
        for (int rb = ra + 1; rb < static_cast<int>(routes.size()); ++rb) {
            const auto& route_a = routes[static_cast<std::size_t>(ra)];
            const auto& route_b = routes[static_cast<std::size_t>(rb)];

            if (static_cast<int>(route_a.size()) > max_route_size || static_cast<int>(route_b.size()) > max_route_size) {
                continue;
            }

            const double base_cost = route_cost(route_a, instance) + route_cost(route_b, instance);
            const auto subsets_a = subset_index_tuples(static_cast<int>(route_a.size()), lam);
            const auto subsets_b = subset_index_tuples(static_cast<int>(route_b.size()), lam);

            int tested = 0;
            for (const auto& idx_a : subsets_a) {
                for (const auto& idx_b : subsets_b) {
                    if (idx_a.empty() && idx_b.empty()) {
                        continue;
                    }
                    ++tested;
                    if (tested > max_pairs_per_routes) {
                        break;
                    }

                    Route picked_a;
                    Route rem_a;
                    Route picked_b;
                    Route rem_b;
                    remove_by_indices(route_a, idx_a, &picked_a, &rem_a);
                    remove_by_indices(route_b, idx_b, &picked_b, &rem_b);

                    if (static_cast<int>(picked_a.size()) > lam || static_cast<int>(picked_b.size()) > lam) {
                        continue;
                    }

                    for (int rev_a = 0; rev_a < 2; ++rev_a) {
                        Route ins_a = picked_b;
                        if (rev_a) {
                            std::reverse(ins_a.begin(), ins_a.end());
                        }

                        for (int rev_b = 0; rev_b < 2; ++rev_b) {
                            Route ins_b = picked_a;
                            if (rev_b) {
                                std::reverse(ins_b.begin(), ins_b.end());
                            }

                            for (int pos_a = 0; pos_a <= static_cast<int>(rem_a.size()); ++pos_a) {
                                Route cand_a;
                                cand_a.reserve(rem_a.size() + ins_a.size());
                                cand_a.insert(cand_a.end(), rem_a.begin(), rem_a.begin() + pos_a);
                                cand_a.insert(cand_a.end(), ins_a.begin(), ins_a.end());
                                cand_a.insert(cand_a.end(), rem_a.begin() + pos_a, rem_a.end());

                                if (!is_feasible_partial_route(cand_a, instance)) {
                                    continue;
                                }

                                for (int pos_b = 0; pos_b <= static_cast<int>(rem_b.size()); ++pos_b) {
                                    Route cand_b;
                                    cand_b.reserve(rem_b.size() + ins_b.size());
                                    cand_b.insert(cand_b.end(), rem_b.begin(), rem_b.begin() + pos_b);
                                    cand_b.insert(cand_b.end(), ins_b.begin(), ins_b.end());
                                    cand_b.insert(cand_b.end(), rem_b.begin() + pos_b, rem_b.end());

                                    if (!is_feasible_partial_route(cand_b, instance)) {
                                        continue;
                                    }

                                    const double cand_cost = route_cost(cand_a, instance) + route_cost(cand_b, instance);
                                    const double delta = cand_cost - base_cost;
                                    if (delta < best_delta) {
                                        Routes candidate = routes;
                                        candidate[static_cast<std::size_t>(ra)] = cand_a;
                                        candidate[static_cast<std::size_t>(rb)] = cand_b;
                                        best_delta = delta;
                                        best_routes = cleanup_routes(candidate);
                                    }
                                }
                            }
                        }
                    }
                }
                if (tested > max_pairs_per_routes) {
                    break;
                }
            }
        }
    }

    return {best_delta, best_routes};
}

Routes iterate_solution_improvement(
    const Routes& routes,
    const std::function<std::pair<double, Routes>(const Routes&)>& move_fn,
    int max_iterations
) {
    Routes current = routes;
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto [delta, candidate] = move_fn(current);
        if (delta < -kEps) {
            current = std::move(candidate);
        } else {
            break;
        }
    }
    return current;
}

// Random neighborhood move operators used by SA/Tabu/ILS.
std::optional<Routes> move_intra_2opt_random(const Routes& routes, std::mt19937& rng) {
    Routes out = routes;
    const int ridx = pick_route_index(out, rng, 4);
    if (ridx < 0) {
        return std::nullopt;
    }
    auto& route = out[static_cast<std::size_t>(ridx)];
    const int i = rand_int(rng, 0, static_cast<int>(route.size()) - 3);
    const int j = rand_int(rng, i + 2, static_cast<int>(route.size()) - 1);
    std::reverse(route.begin() + i, route.begin() + j + 1);
    return cleanup_routes(out);
}

std::optional<Routes> move_intra_relocate_random(const Routes& routes, std::mt19937& rng) {
    Routes out = routes;
    const int ridx = pick_route_index(out, rng, 2);
    if (ridx < 0) {
        return std::nullopt;
    }
    auto& route = out[static_cast<std::size_t>(ridx)];
    const int i = rand_int(rng, 0, static_cast<int>(route.size()) - 1);
    const CustomerId node = route[static_cast<std::size_t>(i)];
    route.erase(route.begin() + i);
    const int j = rand_int(rng, 0, static_cast<int>(route.size()));
    route.insert(route.begin() + j, node);
    return cleanup_routes(out);
}

std::optional<Routes> move_intra_swap_random(const Routes& routes, std::mt19937& rng) {
    Routes out = routes;
    const int ridx = pick_route_index(out, rng, 2);
    if (ridx < 0) {
        return std::nullopt;
    }
    auto& route = out[static_cast<std::size_t>(ridx)];
    int i = rand_int(rng, 0, static_cast<int>(route.size()) - 1);
    int j = rand_int(rng, 0, static_cast<int>(route.size()) - 1);
    while (j == i) {
        j = rand_int(rng, 0, static_cast<int>(route.size()) - 1);
    }
    std::swap(route[static_cast<std::size_t>(i)], route[static_cast<std::size_t>(j)]);
    return cleanup_routes(out);
}

std::optional<Routes> move_inter_insert_random(const Routes& routes, std::mt19937& rng) {
    Routes out = routes;
    const int src = pick_route_index(out, rng, 1);
    if (src < 0) {
        return std::nullopt;
    }
    const int dst = pick_route_index(out, rng, 0, src);
    if (dst < 0) {
        return std::nullopt;
    }

    auto& src_route = out[static_cast<std::size_t>(src)];
    auto& dst_route = out[static_cast<std::size_t>(dst)];
    const int i = rand_int(rng, 0, static_cast<int>(src_route.size()) - 1);
    const CustomerId node = src_route[static_cast<std::size_t>(i)];
    src_route.erase(src_route.begin() + i);

    const int j = rand_int(rng, 0, static_cast<int>(dst_route.size()));
    dst_route.insert(dst_route.begin() + j, node);
    return cleanup_routes(out);
}

std::optional<Routes> move_inter_swap_random(const Routes& routes, std::mt19937& rng) {
    Routes out = routes;
    const int a = pick_route_index(out, rng, 1);
    if (a < 0) {
        return std::nullopt;
    }
    const int b = pick_route_index(out, rng, 1, a);
    if (b < 0) {
        return std::nullopt;
    }

    auto& ra = out[static_cast<std::size_t>(a)];
    auto& rb = out[static_cast<std::size_t>(b)];
    const int ia = rand_int(rng, 0, static_cast<int>(ra.size()) - 1);
    const int ib = rand_int(rng, 0, static_cast<int>(rb.size()) - 1);
    std::swap(ra[static_cast<std::size_t>(ia)], rb[static_cast<std::size_t>(ib)]);
    return cleanup_routes(out);
}

std::optional<Routes> move_two_opt_star_random(const Routes& routes, std::mt19937& rng) {
    Routes out = routes;
    const int a = pick_route_index(out, rng, 1);
    if (a < 0) {
        return std::nullopt;
    }
    const int b = pick_route_index(out, rng, 1, a);
    if (b < 0) {
        return std::nullopt;
    }

    auto ra = out[static_cast<std::size_t>(a)];
    auto rb = out[static_cast<std::size_t>(b)];

    const int cut_a = rand_int(rng, 0, static_cast<int>(ra.size()));
    const int cut_b = rand_int(rng, 0, static_cast<int>(rb.size()));

    Route new_a(ra.begin(), ra.begin() + cut_a);
    new_a.insert(new_a.end(), rb.begin() + cut_b, rb.end());

    Route new_b(rb.begin(), rb.begin() + cut_b);
    new_b.insert(new_b.end(), ra.begin() + cut_a, ra.end());

    out[static_cast<std::size_t>(a)] = std::move(new_a);
    out[static_cast<std::size_t>(b)] = std::move(new_b);
    return cleanup_routes(out);
}

std::optional<Routes> move_cross_random(const Routes& routes, std::mt19937& rng, int lam) {
    Routes out = routes;
    const int a = pick_route_index(out, rng, 0);
    if (a < 0) {
        return std::nullopt;
    }
    const int b = pick_route_index(out, rng, 0, a);
    if (b < 0) {
        return std::nullopt;
    }

    const auto& ra = out[static_cast<std::size_t>(a)];
    const auto& rb = out[static_cast<std::size_t>(b)];

    const int max_a = std::min(lam, static_cast<int>(ra.size()));
    const int max_b = std::min(lam, static_cast<int>(rb.size()));
    const int len_a = rand_int(rng, 0, max_a);
    const int len_b = rand_int(rng, 0, max_b);

    if (len_a == 0 && len_b == 0) {
        return std::nullopt;
    }

    const int start_a = (len_a > 0) ? rand_int(rng, 0, static_cast<int>(ra.size()) - len_a)
                                    : rand_int(rng, 0, static_cast<int>(ra.size()));
    const int start_b = (len_b > 0) ? rand_int(rng, 0, static_cast<int>(rb.size()) - len_b)
                                    : rand_int(rng, 0, static_cast<int>(rb.size()));

    Route seg_a(ra.begin() + start_a, ra.begin() + start_a + len_a);
    Route seg_b(rb.begin() + start_b, rb.begin() + start_b + len_b);

    Route rem_a;
    rem_a.reserve(ra.size() - len_a);
    rem_a.insert(rem_a.end(), ra.begin(), ra.begin() + start_a);
    rem_a.insert(rem_a.end(), ra.begin() + start_a + len_a, ra.end());

    Route rem_b;
    rem_b.reserve(rb.size() - len_b);
    rem_b.insert(rem_b.end(), rb.begin(), rb.begin() + start_b);
    rem_b.insert(rem_b.end(), rb.begin() + start_b + len_b, rb.end());

    Route new_a;
    new_a.reserve(rem_a.size() + seg_b.size());
    new_a.insert(new_a.end(), rem_a.begin(), rem_a.begin() + start_a);
    new_a.insert(new_a.end(), seg_b.begin(), seg_b.end());
    new_a.insert(new_a.end(), rem_a.begin() + start_a, rem_a.end());

    Route new_b;
    new_b.reserve(rem_b.size() + seg_a.size());
    new_b.insert(new_b.end(), rem_b.begin(), rem_b.begin() + start_b);
    new_b.insert(new_b.end(), seg_a.begin(), seg_a.end());
    new_b.insert(new_b.end(), rem_b.begin() + start_b, rem_b.end());

    out[static_cast<std::size_t>(a)] = std::move(new_a);
    out[static_cast<std::size_t>(b)] = std::move(new_b);
    return cleanup_routes(out);
}

std::optional<Routes> move_lambda_interchange_random(const Routes& routes, std::mt19937& rng, int lam) {
    Routes out = routes;
    const int a = pick_route_index(out, rng, 1);
    if (a < 0) {
        return std::nullopt;
    }
    const int b = pick_route_index(out, rng, 1, a);
    if (b < 0) {
        return std::nullopt;
    }

    const auto& ra = out[static_cast<std::size_t>(a)];
    const auto& rb = out[static_cast<std::size_t>(b)];

    const int size_a = rand_int(rng, 0, std::min(lam, static_cast<int>(ra.size())));
    const int size_b = rand_int(rng, 0, std::min(lam, static_cast<int>(rb.size())));
    if (size_a == 0 && size_b == 0) {
        return std::nullopt;
    }

    const auto idx_a = sample_indices(static_cast<int>(ra.size()), size_a, rng);
    const auto idx_b = sample_indices(static_cast<int>(rb.size()), size_b, rng);

    Route picked_a;
    Route rem_a;
    Route picked_b;
    Route rem_b;
    remove_by_indices(ra, idx_a, &picked_a, &rem_a);
    remove_by_indices(rb, idx_b, &picked_b, &rem_b);

    if (rand01(rng) < 0.5) {
        std::reverse(picked_a.begin(), picked_a.end());
    }
    if (rand01(rng) < 0.5) {
        std::reverse(picked_b.begin(), picked_b.end());
    }

    const int pos_a = rand_int(rng, 0, static_cast<int>(rem_a.size()));
    const int pos_b = rand_int(rng, 0, static_cast<int>(rem_b.size()));

    Route new_a;
    new_a.reserve(rem_a.size() + picked_b.size());
    new_a.insert(new_a.end(), rem_a.begin(), rem_a.begin() + pos_a);
    new_a.insert(new_a.end(), picked_b.begin(), picked_b.end());
    new_a.insert(new_a.end(), rem_a.begin() + pos_a, rem_a.end());

    Route new_b;
    new_b.reserve(rem_b.size() + picked_a.size());
    new_b.insert(new_b.end(), rem_b.begin(), rem_b.begin() + pos_b);
    new_b.insert(new_b.end(), picked_a.begin(), picked_a.end());
    new_b.insert(new_b.end(), rem_b.begin() + pos_b, rem_b.end());

    out[static_cast<std::size_t>(a)] = std::move(new_a);
    out[static_cast<std::size_t>(b)] = std::move(new_b);
    return cleanup_routes(out);
}

std::optional<Routes> apply_move_name(
    const std::string& move_name,
    const Routes& routes,
    std::mt19937& rng,
    int lam
) {
    if (move_name == "intra_2opt") {
        return move_intra_2opt_random(routes, rng);
    }
    if (move_name == "intra_relocate") {
        return move_intra_relocate_random(routes, rng);
    }
    if (move_name == "intra_swap") {
        return move_intra_swap_random(routes, rng);
    }
    if (move_name == "inter_insert") {
        return move_inter_insert_random(routes, rng);
    }
    if (move_name == "inter_swap") {
        return move_inter_swap_random(routes, rng);
    }
    if (move_name == "two_opt_star") {
        return move_two_opt_star_random(routes, rng);
    }
    if (move_name == "cross") {
        return move_cross_random(routes, rng, lam);
    }
    if (move_name == "lambda_interchange") {
        return move_lambda_interchange_random(routes, rng, lam);
    }
    return std::nullopt;
}

void validate_move_names(const std::vector<std::string>& moves) {
    static const std::unordered_set<std::string> known = {
        "intra_2opt",
        "intra_relocate",
        "intra_swap",
        "inter_insert",
        "inter_swap",
        "two_opt_star",
        "cross",
        "lambda_interchange",
    };
    for (const auto& move : moves) {
        if (!known.count(move)) {
            throw std::runtime_error("Unknown move: " + move);
        }
    }
}

Routes kick_solution(
    const Routes& routes,
    const std::vector<std::string>& moves,
    std::mt19937& rng,
    int lam,
    int kick_strength,
    const Instance& instance
) {
    Routes kicked = routes;
    for (int i = 0; i < kick_strength; ++i) {
        const auto& move_name = random_choice(moves, rng);
        auto candidate = apply_move_name(move_name, kicked, rng, lam);
        if (!candidate.has_value()) {
            continue;
        }
        if (feasible_solution(*candidate, instance)) {
            kicked = *candidate;
        }
    }
    return kicked;
}

std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (const char ch : input) {
        switch (ch) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << ch;
                break;
        }
    }
    return oss.str();
}

int roulette_index(const std::vector<double>& weights, std::mt19937& rng) {
    double total = 0.0;
    for (double w : weights) {
        total += std::max(0.0, w);
    }
    if (total <= 0.0) {
        return rand_int(rng, 0, static_cast<int>(weights.size()) - 1);
    }

    const double ticket = rand01(rng) * total;
    double acc = 0.0;
    for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
        acc += std::max(0.0, weights[static_cast<std::size_t>(i)]);
        if (ticket <= acc) {
            return i;
        }
    }
    return static_cast<int>(weights.size()) - 1;
}

double default_initial_temperature(double cost) {
    return std::max(1.0, 0.05 * std::max(1.0, cost));
}

Routes remove_nodes(const Routes& routes, const std::unordered_set<CustomerId>& node_set) {
    Routes out;
    out.reserve(routes.size());
    for (const auto& route : routes) {
        Route reduced;
        reduced.reserve(route.size());
        for (CustomerId node : route) {
            if (!node_set.count(node)) {
                reduced.push_back(node);
            }
        }
        if (!reduced.empty()) {
            out.push_back(std::move(reduced));
        }
    }
    return out;
}

std::pair<Routes, std::vector<CustomerId>> destroy_random(const Routes& routes, int q, std::mt19937& rng) {
    auto all_nodes = flatten_routes(routes);
    if (all_nodes.empty()) {
        return {routes, {}};
    }

    q = std::min(q, static_cast<int>(all_nodes.size()));
    std::shuffle(all_nodes.begin(), all_nodes.end(), rng);
    std::vector<CustomerId> removed(all_nodes.begin(), all_nodes.begin() + q);

    std::unordered_set<CustomerId> removed_set(removed.begin(), removed.end());
    Routes partial = remove_nodes(routes, removed_set);
    return {partial, removed};
}

std::pair<Routes, std::vector<CustomerId>> destroy_related(
    const Routes& routes,
    int q,
    std::mt19937& rng,
    const Instance& instance
) {
    auto all_nodes = flatten_routes(routes);
    if (all_nodes.empty()) {
        return {routes, {}};
    }

    q = std::min(q, static_cast<int>(all_nodes.size()));
    const CustomerId seed = random_choice(all_nodes, rng);

    std::unordered_set<CustomerId> remaining(all_nodes.begin(), all_nodes.end());
    std::vector<CustomerId> removed;
    removed.reserve(static_cast<std::size_t>(q));
    removed.push_back(seed);
    remaining.erase(seed);

    while (static_cast<int>(removed.size()) < q && !remaining.empty()) {
        CustomerId best_node = *remaining.begin();
        double best_dist = std::numeric_limits<double>::infinity();

        for (CustomerId node : remaining) {
            double d = std::numeric_limits<double>::infinity();
            for (CustomerId r : removed) {
                d = std::min(d, euclidean(customer_point(instance, node), customer_point(instance, r)));
            }
            if (d < best_dist) {
                best_dist = d;
                best_node = node;
            }
        }

        removed.push_back(best_node);
        remaining.erase(best_node);
    }

    std::unordered_set<CustomerId> removed_set(removed.begin(), removed.end());
    return {remove_nodes(routes, removed_set), removed};
}

double removal_gain(const Route& route, int pos, const Instance& instance) {
    const CustomerId node = route[static_cast<std::size_t>(pos)];
    const Point prev = (pos == 0) ? instance.depot : customer_point(instance, route[static_cast<std::size_t>(pos - 1)]);
    const Point curr = customer_point(instance, node);
    const Point next = (pos == static_cast<int>(route.size()) - 1)
        ? instance.depot
        : customer_point(instance, route[static_cast<std::size_t>(pos + 1)]);

    const double old_cost = euclidean(prev, curr) + euclidean(curr, next);
    const double new_cost = euclidean(prev, next);
    return old_cost - new_cost;
}

std::pair<Routes, std::vector<CustomerId>> destroy_worst(
    const Routes& routes,
    int q,
    std::mt19937& rng,
    const Instance& instance
) {
    std::vector<std::pair<double, CustomerId>> candidates;
    for (const auto& route : routes) {
        for (int pos = 0; pos < static_cast<int>(route.size()); ++pos) {
            const CustomerId node = route[static_cast<std::size_t>(pos)];
            double gain = removal_gain(route, pos, instance);
            gain += rand01(rng) * 1e-4;
            candidates.emplace_back(gain, node);
        }
    }

    if (candidates.empty()) {
        return {routes, {}};
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    q = std::min(q, static_cast<int>(candidates.size()));
    std::vector<CustomerId> removed;
    removed.reserve(static_cast<std::size_t>(q));
    for (int i = 0; i < q; ++i) {
        removed.push_back(candidates[static_cast<std::size_t>(i)].second);
    }

    std::unordered_set<CustomerId> removed_set(removed.begin(), removed.end());
    return {remove_nodes(routes, removed_set), removed};
}

double insertion_delta(const Route& route, int pos, CustomerId node, const Instance& instance) {
    const Point prev = (pos == 0) ? instance.depot : customer_point(instance, route[static_cast<std::size_t>(pos - 1)]);
    const Point next = (pos == static_cast<int>(route.size()))
        ? instance.depot
        : customer_point(instance, route[static_cast<std::size_t>(pos)]);
    const Point np = customer_point(instance, node);

    return euclidean(prev, np) + euclidean(np, next) - euclidean(prev, next);
}

std::vector<std::tuple<double, int, int>> best_insertions_for_node(
    CustomerId node,
    const Routes& routes,
    const Instance& instance
) {
    std::vector<std::tuple<double, int, int>> insertions;

    for (int ridx = 0; ridx < static_cast<int>(routes.size()); ++ridx) {
        const auto& route = routes[static_cast<std::size_t>(ridx)];
        if (route_load(route, instance) + customer_demand(instance, node) > instance.capacity + kEps) {
            continue;
        }
        for (int pos = 0; pos <= static_cast<int>(route.size()); ++pos) {
            const double delta = insertion_delta(route, pos, node, instance);
            insertions.emplace_back(delta, ridx, pos);
        }
    }

    const double new_route_delta = 2.0 * euclidean(instance.depot, customer_point(instance, node));
    insertions.emplace_back(new_route_delta, static_cast<int>(routes.size()), 0);

    std::sort(insertions.begin(), insertions.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });

    return insertions;
}

void apply_insertion(Routes* routes, CustomerId node, int ridx, int pos) {
    if (ridx == static_cast<int>(routes->size())) {
        routes->push_back({node});
    } else {
        auto& route = (*routes)[static_cast<std::size_t>(ridx)];
        route.insert(route.begin() + pos, node);
    }
}

std::optional<Routes> repair_greedy(
    const Routes& partial_routes,
    const std::vector<CustomerId>& removed_nodes,
    const Instance& instance
) {
    Routes routes = partial_routes;
    std::vector<CustomerId> pool = removed_nodes;

    while (!pool.empty()) {
        std::optional<std::tuple<double, CustomerId, int, int>> best;

        for (CustomerId node : pool) {
            auto insertions = best_insertions_for_node(node, routes, instance);
            if (insertions.empty()) {
                continue;
            }
            const auto& first = insertions.front();
            const auto cand = std::make_tuple(std::get<0>(first), node, std::get<1>(first), std::get<2>(first));
            if (!best.has_value() || std::get<0>(cand) < std::get<0>(*best)) {
                best = cand;
            }
        }

        if (!best.has_value()) {
            return std::nullopt;
        }

        const CustomerId node = std::get<1>(*best);
        apply_insertion(&routes, node, std::get<2>(*best), std::get<3>(*best));

        auto it = std::find(pool.begin(), pool.end(), node);
        if (it != pool.end()) {
            pool.erase(it);
        }
    }

    return cleanup_routes(routes);
}

std::optional<Routes> repair_regret2(
    const Routes& partial_routes,
    const std::vector<CustomerId>& removed_nodes,
    const Instance& instance
) {
    Routes routes = partial_routes;
    std::vector<CustomerId> pool = removed_nodes;

    while (!pool.empty()) {
        std::optional<std::tuple<double, CustomerId, int, int>> chosen;

        for (CustomerId node : pool) {
            auto insertions = best_insertions_for_node(node, routes, instance);
            if (insertions.empty()) {
                continue;
            }

            const double best_delta = std::get<0>(insertions[0]);
            const double second_delta = (insertions.size() > 1) ? std::get<0>(insertions[1]) : best_delta + 1e6;
            const double regret = second_delta - best_delta;
            const int ridx = std::get<1>(insertions[0]);
            const int pos = std::get<2>(insertions[0]);

            const auto cand = std::make_tuple(regret, node, ridx, pos);
            if (!chosen.has_value() || std::get<0>(cand) > std::get<0>(*chosen)) {
                chosen = cand;
            }
        }

        if (!chosen.has_value()) {
            return std::nullopt;
        }

        const CustomerId node = std::get<1>(*chosen);
        apply_insertion(&routes, node, std::get<2>(*chosen), std::get<3>(*chosen));

        auto it = std::find(pool.begin(), pool.end(), node);
        if (it != pool.end()) {
            pool.erase(it);
        }
    }

    return cleanup_routes(routes);
}

Routes intensify_with_local_search(
    const Instance& instance,
    const Routes& routes,
    bool use_local_search,
    int lam
) {
    if (!use_local_search) {
        return routes;
    }

    IntraImproveOptions intra;
    intra.methods = {"2opt", "or_opt", "relocate", "exchange"};
    intra.max_passes = 1;

    InterImproveOptions inter;
    inter.methods = {"insert", "swap", "cross"};
    inter.max_passes = 1;
    inter.lam = lam;

    Routes current = improve_intra(instance, routes, intra);
    current = improve_inter(instance, current, inter);
    return current;
}

std::vector<CustomerId> routes_to_permutation(const Routes& routes) {
    return flatten_routes(routes);
}

Routes decode_permutation(
    const std::vector<CustomerId>& perm,
    const Instance& instance
) {
    Routes routes;
    Route route;
    double load = 0.0;

    for (CustomerId node : perm) {
        const double d = customer_demand(instance, node);
        if (load + d <= instance.capacity + kEps) {
            route.push_back(node);
            load += d;
        } else {
            if (!route.empty()) {
                routes.push_back(route);
            }
            route = {node};
            load = d;
        }
    }

    if (!route.empty()) {
        routes.push_back(route);
    }

    return routes;
}

bool valid_permutation(
    const std::vector<CustomerId>& perm,
    const std::unordered_set<CustomerId>& customer_set
) {
    if (perm.size() != customer_set.size()) {
        return false;
    }
    std::unordered_set<CustomerId> seen(perm.begin(), perm.end());
    return seen == customer_set;
}

std::pair<double, Routes> evaluate_permutation(
    const std::vector<CustomerId>& perm,
    const Instance& instance
) {
    Routes routes = decode_permutation(perm, instance);
    return {solution_cost(routes, instance), routes};
}

std::vector<CustomerId> tournament_select(
    const std::vector<std::vector<CustomerId>>& population,
    const std::vector<double>& fitness,
    int k,
    std::mt19937& rng
) {
    const int take = std::min(k, static_cast<int>(population.size()));
    std::vector<int> idx(population.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(static_cast<std::size_t>(take));

    int best = idx[0];
    for (int i : idx) {
        if (fitness[static_cast<std::size_t>(i)] < fitness[static_cast<std::size_t>(best)]) {
            best = i;
        }
    }
    return population[static_cast<std::size_t>(best)];
}

std::vector<CustomerId> order_crossover(
    const std::vector<CustomerId>& parent_a,
    const std::vector<CustomerId>& parent_b,
    std::mt19937& rng
) {
    const int n = static_cast<int>(parent_a.size());
    if (n < 2) {
        return parent_a;
    }

    int i = rand_int(rng, 0, n - 1);
    int j = rand_int(rng, 0, n - 1);
    if (i > j) {
        std::swap(i, j);
    }

    std::vector<CustomerId> child(static_cast<std::size_t>(n), -1);
    std::unordered_set<CustomerId> taken;

    for (int pos = i; pos <= j; ++pos) {
        child[static_cast<std::size_t>(pos)] = parent_a[static_cast<std::size_t>(pos)];
        taken.insert(parent_a[static_cast<std::size_t>(pos)]);
    }

    int ptr = 0;
    for (int pos = 0; pos < n; ++pos) {
        if (child[static_cast<std::size_t>(pos)] != -1) {
            continue;
        }
        while (ptr < n && taken.count(parent_b[static_cast<std::size_t>(ptr)])) {
            ++ptr;
        }
        child[static_cast<std::size_t>(pos)] = parent_b[static_cast<std::size_t>(ptr)];
        ++ptr;
    }

    return child;
}

void mutate(std::vector<CustomerId>* perm, std::mt19937& rng) {
    const int n = static_cast<int>(perm->size());
    if (n < 2) {
        return;
    }

    const int op = rand_int(rng, 0, 2);
    if (op == 0) {
        int i = rand_int(rng, 0, n - 1);
        int j = rand_int(rng, 0, n - 1);
        while (j == i) {
            j = rand_int(rng, 0, n - 1);
        }
        std::swap((*perm)[static_cast<std::size_t>(i)], (*perm)[static_cast<std::size_t>(j)]);
        return;
    }

    if (op == 1) {
        int i = rand_int(rng, 0, n - 1);
        int j = rand_int(rng, 0, n - 1);
        if (i > j) {
            std::swap(i, j);
        }
        std::reverse(perm->begin() + i, perm->begin() + j + 1);
        return;
    }

    int i = rand_int(rng, 0, n - 1);
    int j = rand_int(rng, 0, n - 1);
    CustomerId node = (*perm)[static_cast<std::size_t>(i)];
    perm->erase(perm->begin() + i);
    perm->insert(perm->begin() + j, node);
}

std::vector<std::vector<CustomerId>> seed_population(
    const Instance& instance,
    int population_size,
    std::mt19937& rng,
    int duplicate_tolerance_multiplier
) {
    std::vector<std::vector<CustomerId>> population;
    std::unordered_set<std::string> seen;

    auto try_add = [&](const std::vector<CustomerId>& perm) {
        const std::string key = permutation_key(perm);
        if (seen.count(key)) {
            return false;
        }
        population.push_back(perm);
        seen.insert(key);
        return true;
    };

    try {
        try_add(routes_to_permutation(nearest_neighbor(instance)));
    } catch (...) {
    }
    try {
        try_add(routes_to_permutation(clarke_wright(instance)));
    } catch (...) {
    }
    try {
        try_add(routes_to_permutation(sweep(instance)));
    } catch (...) {
    }
    try {
        try_add(routes_to_permutation(cheapest_insertion(instance)));
    } catch (...) {
    }

    auto genes = instance.customer_ids;

    while (static_cast<int>(population.size()) < population_size) {
        auto perm = genes;
        std::shuffle(perm.begin(), perm.end(), rng);
        const bool added = try_add(perm);
        if (!added && static_cast<int>(population.size()) < population_size
            && static_cast<int>(seen.size()) > duplicate_tolerance_multiplier * population_size) {
            population.push_back(perm);
        }
    }

    if (static_cast<int>(population.size()) > population_size) {
        population.resize(static_cast<std::size_t>(population_size));
    }

    return population;
}

Routes local_search_round(
    const Instance& instance,
    const Routes& routes,
    int lam,
    const std::vector<std::string>& intra_methods,
    const std::vector<std::string>& inter_methods,
    int rounds
) {
    Routes current = routes;
    for (int i = 0; i < std::max(1, rounds); ++i) {
        const double before = solution_cost(current, instance);

        IntraImproveOptions intra;
        intra.methods = intra_methods;
        intra.max_passes = 1;

        InterImproveOptions inter;
        inter.methods = inter_methods;
        inter.max_passes = 1;
        inter.lam = lam;

        current = improve_intra(instance, current, intra);
        current = improve_inter(instance, current, inter);

        const double after = solution_cost(current, instance);
        if (after >= before - kEps) {
            break;
        }
    }
    return current;
}

Routes perturb_solution(
    const Routes& routes,
    int perturbation_strength,
    const std::vector<std::string>& perturbation_moves,
    std::mt19937& rng,
    int lam,
    const Instance& instance
) {
    Routes perturbed = routes;
    const int moves_to_apply = std::max(1, perturbation_strength);

    for (int i = 0; i < moves_to_apply; ++i) {
        const auto& move_name = random_choice(perturbation_moves, rng);
        auto candidate = apply_move_name(move_name, perturbed, rng, lam);
        if (!candidate.has_value()) {
            continue;
        }
        if (feasible_solution(*candidate, instance)) {
            perturbed = *candidate;
        }
    }

    return perturbed;
}

std::vector<std::vector<double>> build_distance_matrix(const Instance& instance, const std::vector<CustomerId>& customer_ids) {
    std::vector<Point> points;
    points.reserve(customer_ids.size() + 1);
    points.push_back(instance.depot);
    for (CustomerId cid : customer_ids) {
        points.push_back(customer_point(instance, cid));
    }

    const int n = static_cast<int>(points.size());
    std::vector<std::vector<double>> dist(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double d = euclidean(points[static_cast<std::size_t>(i)], points[static_cast<std::size_t>(j)]);
            dist[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = d;
            dist[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = d;
        }
    }

    return dist;
}

std::vector<std::vector<double>> inverse_distance(const std::vector<std::vector<double>>& dist) {
    const int n = static_cast<int>(dist.size());
    std::vector<std::vector<double>> eta(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                eta[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.0;
            } else {
                eta[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 1.0 / std::max(dist[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], 1e-9);
            }
        }
    }

    return eta;
}

CustomerId select_next_customer(
    const std::vector<CustomerId>& feasible,
    const std::vector<double>& scores,
    std::mt19937& rng
) {
    double total = 0.0;
    for (double score : scores) {
        total += score;
    }

    if (total <= 0.0) {
        return random_choice(feasible, rng);
    }

    const double r = rand01(rng) * total;
    double acc = 0.0;
    for (std::size_t i = 0; i < feasible.size(); ++i) {
        acc += scores[i];
        if (acc >= r) {
            return feasible[i];
        }
    }
    return feasible.back();
}

Routes construct_ant_solution(
    const Instance& instance,
    const std::vector<CustomerId>& customer_ids,
    const std::vector<std::vector<double>>& tau,
    const std::vector<std::vector<double>>& eta,
    double alpha,
    double beta,
    double q0,
    std::mt19937& rng
) {
    std::unordered_set<CustomerId> unvisited(customer_ids.begin(), customer_ids.end());
    Routes routes;

    std::unordered_map<CustomerId, int> id_to_internal;
    for (int i = 0; i < static_cast<int>(customer_ids.size()); ++i) {
        id_to_internal[customer_ids[static_cast<std::size_t>(i)]] = i + 1;
    }

    while (!unvisited.empty()) {
        Route route;
        double rem_cap = instance.capacity;
        int current = 0;

        while (true) {
            std::vector<CustomerId> feasible;
            feasible.reserve(unvisited.size());
            for (CustomerId cid : unvisited) {
                if (customer_demand(instance, cid) <= rem_cap + kEps) {
                    feasible.push_back(cid);
                }
            }

            if (feasible.empty()) {
                break;
            }

            std::vector<double> desirability;
            desirability.reserve(feasible.size());
            for (CustomerId cid : feasible) {
                const int nxt = id_to_internal[cid];
                desirability.push_back(
                    std::pow(tau[static_cast<std::size_t>(current)][static_cast<std::size_t>(nxt)], alpha)
                    * std::pow(eta[static_cast<std::size_t>(current)][static_cast<std::size_t>(nxt)], beta)
                );
            }

            CustomerId chosen;
            if (rand01(rng) < q0) {
                std::size_t best_idx = 0;
                for (std::size_t i = 1; i < desirability.size(); ++i) {
                    if (desirability[i] > desirability[best_idx]) {
                        best_idx = i;
                    }
                }
                chosen = feasible[best_idx];
            } else {
                chosen = select_next_customer(feasible, desirability, rng);
            }

            route.push_back(chosen);
            unvisited.erase(chosen);
            rem_cap -= customer_demand(instance, chosen);
            current = id_to_internal[chosen];
        }

        if (route.empty()) {
            throw std::runtime_error("Ant failed to construct a feasible non-empty route");
        }

        routes.push_back(route);
    }

    return cleanup_routes(routes);
}

void deposit_pheromone(
    const Routes& routes,
    double delta_tau,
    std::vector<std::vector<double>>* tau,
    const std::vector<CustomerId>& customer_ids
) {
    std::unordered_map<CustomerId, int> id_to_internal;
    for (int i = 0; i < static_cast<int>(customer_ids.size()); ++i) {
        id_to_internal[customer_ids[static_cast<std::size_t>(i)]] = i + 1;
    }

    for (const auto& route : routes) {
        int prev = 0;
        for (CustomerId cid : route) {
            const int curr = id_to_internal[cid];
            (*tau)[static_cast<std::size_t>(prev)][static_cast<std::size_t>(curr)] += delta_tau;
            (*tau)[static_cast<std::size_t>(curr)][static_cast<std::size_t>(prev)] += delta_tau;
            prev = curr;
        }
        (*tau)[static_cast<std::size_t>(prev)][0] += delta_tau;
        (*tau)[0][static_cast<std::size_t>(prev)] += delta_tau;
    }
}

void bound_pheromone(std::vector<std::vector<double>>* tau, double tau_min, double tau_max) {
    for (auto& row : *tau) {
        for (double& value : row) {
            if (value < tau_min) {
                value = tau_min;
            } else if (value > tau_max) {
                value = tau_max;
            }
        }
    }
}

}  // namespace

Instance load_instance_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open instance file: " + path);
    }

    Instance instance;

    bool has_capacity = false;
    bool has_depot = false;
    bool has_customers = false;

    std::string token;
    while (read_next_token(in, &token)) {
        const std::string key = to_lower(token);
        if (key == "capacity") {
            std::string value;
            if (!read_next_token(in, &value)) {
                throw std::runtime_error("Missing capacity value");
            }
            if (!try_parse_double(value, &instance.capacity)) {
                throw std::runtime_error("Invalid capacity value: " + value);
            }
            has_capacity = true;
            continue;
        }

        if (key == "depot") {
            std::string sx;
            std::string sy;
            if (!read_next_token(in, &sx) || !read_next_token(in, &sy)) {
                throw std::runtime_error("Invalid depot line; expected: depot <x> <y>");
            }
            if (!try_parse_double(sx, &instance.depot.x) || !try_parse_double(sy, &instance.depot.y)) {
                throw std::runtime_error("Invalid depot coordinates");
            }
            has_depot = true;
            continue;
        }

        if (key == "customers") {
            std::string n_str;
            if (!read_next_token(in, &n_str)) {
                throw std::runtime_error("Missing customer count after 'customers'");
            }
            int n = 0;
            if (!try_parse_int(n_str, &n) || n < 0) {
                throw std::runtime_error("Invalid customer count: " + n_str);
            }

            for (int i = 0; i < n; ++i) {
                std::string sid;
                std::string sx;
                std::string sy;
                std::string sd;
                if (!read_next_token(in, &sid) || !read_next_token(in, &sx) || !read_next_token(in, &sy)
                    || !read_next_token(in, &sd)) {
                    throw std::runtime_error("Incomplete customer row at index " + std::to_string(i));
                }
                int cid = 0;
                Point p;
                double d = 0.0;
                if (!try_parse_int(sid, &cid) || !try_parse_double(sx, &p.x) || !try_parse_double(sy, &p.y)
                    || !try_parse_double(sd, &d)) {
                    throw std::runtime_error("Invalid customer row values at index " + std::to_string(i));
                }

                instance.customer_ids.push_back(cid);
                instance.coords[cid] = p;
                instance.demand[cid] = d;
            }

            has_customers = true;
            continue;
        }

        throw std::runtime_error("Unknown token in instance file: " + token);
    }

    if (!has_capacity || !has_depot || !has_customers) {
        throw std::runtime_error("Instance file is missing required sections: capacity/depot/customers");
    }

    ensure_valid_instance(instance);
    return instance;
}

Routes load_routes_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open routes file: " + path);
    }

    Routes routes;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string first;
        if (!(iss >> first)) {
            continue;
        }

        if (to_lower(first) == "routes") {
            continue;
        }

        int value = 0;
        if (!try_parse_int(first, &value)) {
            continue;
        }

        Route route;
        route.push_back(value);

        std::string token;
        while (iss >> token) {
            int cid = 0;
            if (!try_parse_int(token, &cid)) {
                throw std::runtime_error("Invalid route token in routes file: " + token);
            }
            route.push_back(cid);
        }

        if (!route.empty()) {
            routes.push_back(route);
        }
    }

    return cleanup_routes(routes);
}

void ensure_valid_instance(const Instance& instance) {
    if (instance.capacity <= 0.0) {
        throw std::runtime_error("Vehicle capacity must be > 0");
    }
    if (instance.customer_ids.empty()) {
        throw std::runtime_error("Instance must include at least one customer");
    }

    std::unordered_set<CustomerId> unique(instance.customer_ids.begin(), instance.customer_ids.end());
    if (unique.size() != instance.customer_ids.size()) {
        throw std::runtime_error("Duplicate customer IDs in instance");
    }

    for (CustomerId cid : instance.customer_ids) {
        const auto d_it = instance.demand.find(cid);
        const auto c_it = instance.coords.find(cid);
        if (d_it == instance.demand.end() || c_it == instance.coords.end()) {
            throw std::runtime_error("Missing coordinate or demand for customer: " + std::to_string(cid));
        }
        if (d_it->second > instance.capacity + kEps) {
            throw std::runtime_error("At least one customer demand exceeds vehicle capacity");
        }
    }
}

double route_cost(const Route& route, const Instance& instance) {
    if (route.empty()) {
        return 0.0;
    }

    double cost = euclidean(instance.depot, customer_point(instance, route.front()));
    for (std::size_t i = 0; i + 1 < route.size(); ++i) {
        cost += euclidean(customer_point(instance, route[i]), customer_point(instance, route[i + 1]));
    }
    cost += euclidean(customer_point(instance, route.back()), instance.depot);
    return cost;
}

double solution_cost(const Routes& routes, const Instance& instance) {
    double total = 0.0;
    for (const auto& route : routes) {
        total += route_cost(route, instance);
    }
    return total;
}

double route_load(const Route& route, const Instance& instance) {
    double load = 0.0;
    for (CustomerId node : route) {
        load += customer_demand(instance, node);
    }
    return load;
}

bool feasible_solution(const Routes& routes, const Instance& instance) {
    const auto flat = flatten_routes(routes);
    if (flat.size() != instance.customer_ids.size()) {
        return false;
    }

    std::unordered_set<CustomerId> seen(flat.begin(), flat.end());
    std::unordered_set<CustomerId> expected(instance.customer_ids.begin(), instance.customer_ids.end());
    if (seen != expected) {
        return false;
    }

    for (const auto& route : routes) {
        if (route_load(route, instance) > instance.capacity + kEps) {
            return false;
        }
    }
    return true;
}

Routes nearest_neighbor(const Instance& instance) {
    ensure_valid_instance(instance);

    std::unordered_set<CustomerId> unvisited(instance.customer_ids.begin(), instance.customer_ids.end());
    Routes routes;

    while (!unvisited.empty()) {
        Route route;
        double load = 0.0;
        Point current = instance.depot;

        while (true) {
            bool found = false;
            CustomerId best_node = -1;
            double best_distance = std::numeric_limits<double>::infinity();

            for (CustomerId cid : unvisited) {
                const double d = customer_demand(instance, cid);
                if (load + d > instance.capacity + kEps) {
                    continue;
                }

                const double distance = euclidean(current, customer_point(instance, cid));
                if (distance < best_distance) {
                    best_distance = distance;
                    best_node = cid;
                    found = true;
                }
            }

            if (!found) {
                break;
            }

            route.push_back(best_node);
            load += customer_demand(instance, best_node);
            current = customer_point(instance, best_node);
            unvisited.erase(best_node);
        }

        if (route.empty()) {
            throw std::runtime_error("Could not build feasible route with current capacity/demands");
        }
        routes.push_back(route);
    }

    return routes;
}

Routes clarke_wright(const Instance& instance) {
    ensure_valid_instance(instance);

    Routes routes;
    routes.reserve(instance.customer_ids.size());
    for (CustomerId cid : instance.customer_ids) {
        routes.push_back({cid});
    }

    struct Saving {
        double value = 0.0;
        CustomerId i = -1;
        CustomerId j = -1;
    };

    std::vector<Saving> savings;
    for (std::size_t a = 0; a < instance.customer_ids.size(); ++a) {
        for (std::size_t b = a + 1; b < instance.customer_ids.size(); ++b) {
            const CustomerId i = instance.customer_ids[a];
            const CustomerId j = instance.customer_ids[b];

            const double distance_ij = euclidean(customer_point(instance, i), customer_point(instance, j));
            const double distance_id = euclidean(instance.depot, customer_point(instance, i));
            const double distance_jd = euclidean(instance.depot, customer_point(instance, j));
            savings.push_back({distance_id + distance_jd - distance_ij, i, j});
        }
    }

    std::sort(savings.begin(), savings.end(), [](const Saving& a, const Saving& b) {
        return a.value > b.value;
    });

    for (const auto& saving : savings) {
        const int idx_i = find_route_with_customer(routes, saving.i);
        const int idx_j = find_route_with_customer(routes, saving.j);

        if (idx_i < 0 || idx_j < 0 || idx_i == idx_j) {
            continue;
        }

        const auto& route_i = routes[static_cast<std::size_t>(idx_i)];
        const auto& route_j = routes[static_cast<std::size_t>(idx_j)];

        const double load_i = route_load(route_i, instance);
        const double load_j = route_load(route_j, instance);

        const bool i_on_end = (!route_i.empty() && (route_i.front() == saving.i || route_i.back() == saving.i));
        const bool j_on_end = (!route_j.empty() && (route_j.front() == saving.j || route_j.back() == saving.j));

        if (load_i + load_j > instance.capacity + kEps || !i_on_end || !j_on_end) {
            continue;
        }

        Route route_i_oriented = route_i;
        if (!route_i_oriented.empty() && route_i_oriented.front() == saving.i) {
            std::reverse(route_i_oriented.begin(), route_i_oriented.end());
        }

        Route route_j_oriented = route_j;
        if (!route_j_oriented.empty() && route_j_oriented.back() == saving.j) {
            std::reverse(route_j_oriented.begin(), route_j_oriented.end());
        }

        Route merged = route_i_oriented;
        merged.insert(merged.end(), route_j_oriented.begin(), route_j_oriented.end());

        if (idx_i > idx_j) {
            routes.erase(routes.begin() + idx_i);
            routes.erase(routes.begin() + idx_j);
        } else {
            routes.erase(routes.begin() + idx_j);
            routes.erase(routes.begin() + idx_i);
        }
        routes.push_back(std::move(merged));
    }

    return routes;
}

Routes cheapest_insertion(const Instance& instance) {
    ensure_valid_instance(instance);

    Routes routes = {Route{}};
    std::unordered_set<CustomerId> unvisited(instance.customer_ids.begin(), instance.customer_ids.end());

    while (!unvisited.empty()) {
        bool found = false;
        CustomerId best_customer = -1;
        int best_route_idx = -1;
        int best_position = -1;
        double best_cost = std::numeric_limits<double>::infinity();

        for (CustomerId customer : unvisited) {
            for (int ridx = 0; ridx < static_cast<int>(routes.size()); ++ridx) {
                const auto& route = routes[static_cast<std::size_t>(ridx)];

                if (route_load(route, instance) + customer_demand(instance, customer) > instance.capacity + kEps) {
                    continue;
                }

                for (int pos = 0; pos <= static_cast<int>(route.size()); ++pos) {
                    Route candidate = route;
                    candidate.insert(candidate.begin() + pos, customer);
                    const double cost = route_cost(candidate, instance);
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_customer = customer;
                        best_route_idx = ridx;
                        best_position = pos;
                        found = true;
                    }
                }
            }
        }

        if (!found) {
            routes.push_back(Route{});
            continue;
        }

        auto& best_route = routes[static_cast<std::size_t>(best_route_idx)];
        best_route.insert(best_route.begin() + best_position, best_customer);
        unvisited.erase(best_customer);
    }

    return cleanup_routes(routes);
}

Routes sweep(const Instance& instance) {
    ensure_valid_instance(instance);

    std::vector<std::pair<double, CustomerId>> angles;
    angles.reserve(instance.customer_ids.size());

    for (CustomerId cid : instance.customer_ids) {
        const Point p = customer_point(instance, cid);
        const double angle = std::atan2(p.y - instance.depot.y, p.x - instance.depot.x);
        angles.emplace_back(angle, cid);
    }

    std::sort(angles.begin(), angles.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    Routes routes;
    Route current;
    double load = 0.0;

    for (const auto& [_, cid] : angles) {
        const double d = customer_demand(instance, cid);
        if (load + d <= instance.capacity + kEps) {
            current.push_back(cid);
            load += d;
        } else {
            if (!current.empty()) {
                routes.push_back(current);
            }
            current = {cid};
            load = d;
        }
    }

    if (!current.empty()) {
        routes.push_back(current);
    }

    return routes;
}

Routes improve_intra(const Instance& instance, const Routes& routes, const IntraImproveOptions& options) {
    Routes current = routes;
    double current_cost = solution_cost(current, instance);

    auto apply_method = [&](const std::string& method, const Routes& in) {
        Routes out = in;

        if (method == "2opt") {
            for (auto& route : out) {
                route = iterate_route_improvement(
                    route,
                    [&](const Route& r) { return best_two_opt_move(r, instance); },
                    options.max_iterations
                );
            }
            return out;
        }

        if (method == "3opt") {
            for (auto& route : out) {
                if (static_cast<int>(route.size()) > options.three_opt_max_route_size) {
                    continue;
                }
                route = iterate_route_improvement(
                    route,
                    [&](const Route& r) { return best_three_opt_move(r, instance); },
                    std::min(options.max_iterations, 10)
                );
            }
            return out;
        }

        if (method == "or_opt") {
            for (auto& route : out) {
                route = iterate_route_improvement(
                    route,
                    [&](const Route& r) { return best_or_opt_move(r, instance, options.or_opt_segment_lengths); },
                    options.max_iterations
                );
            }
            return out;
        }

        if (method == "relocate") {
            for (auto& route : out) {
                route = iterate_route_improvement(
                    route,
                    [&](const Route& r) { return best_relocate_move(r, instance); },
                    options.max_iterations
                );
            }
            return out;
        }

        if (method == "exchange") {
            for (auto& route : out) {
                route = iterate_route_improvement(
                    route,
                    [&](const Route& r) { return best_exchange_move(r, instance); },
                    options.max_iterations
                );
            }
            return out;
        }

        if (method == "geni") {
            for (auto& route : out) {
                route = iterate_route_improvement(
                    route,
                    [&](const Route& r) { return best_geni_move(r, instance, options.nearest_k); },
                    options.max_iterations
                );
            }
            return out;
        }

        throw std::runtime_error("Unknown improve_intra method: " + method);
    };

    for (int pass = 0; pass < options.max_passes; ++pass) {
        bool improved = false;

        for (const auto& method : options.methods) {
            const Routes candidate = apply_method(method, current);
            const double candidate_cost = solution_cost(candidate, instance);
            if (candidate_cost + kEps < current_cost) {
                current = candidate;
                current_cost = candidate_cost;
                improved = true;
            }
        }

        if (!improved) {
            break;
        }
    }

    return current;
}

Routes improve_inter(const Instance& instance, const Routes& routes, const InterImproveOptions& options) {
    Routes current = routes;
    double current_cost = solution_cost(current, instance);

    auto apply_method = [&](const std::string& method, const Routes& in) {
        if (method == "2opt*") {
            return iterate_solution_improvement(
                in,
                [&](const Routes& r) { return best_two_opt_star_move(r, instance); },
                std::min(options.max_iterations, 20)
            );
        }
        if (method == "insert") {
            return iterate_solution_improvement(
                in,
                [&](const Routes& r) { return best_insert_move(r, instance); },
                options.max_iterations
            );
        }
        if (method == "swap") {
            return iterate_solution_improvement(
                in,
                [&](const Routes& r) { return best_swap_move(r, instance); },
                options.max_iterations
            );
        }
        if (method == "cross") {
            return iterate_solution_improvement(
                in,
                [&](const Routes& r) { return best_cross_move(r, instance, options.lam); },
                std::min(options.max_iterations, 20)
            );
        }
        if (method == "lambda_interchange") {
            return iterate_solution_improvement(
                in,
                [&](const Routes& r) {
                    return best_lambda_interchange_move(
                        r,
                        instance,
                        options.lam,
                        options.lambda_max_route_size,
                        options.lambda_max_pairs_per_routes
                    );
                },
                std::min(options.max_iterations, 10)
            );
        }

        throw std::runtime_error("Unknown improve_inter method: " + method);
    };

    for (int pass = 0; pass < options.max_passes; ++pass) {
        bool improved = false;

        for (const auto& method : options.methods) {
            const Routes candidate = apply_method(method, current);
            const double candidate_cost = solution_cost(candidate, instance);
            if (candidate_cost + kEps < current_cost) {
                current = candidate;
                current_cost = candidate_cost;
                improved = true;
            }
        }

        if (!improved) {
            break;
        }
    }

    return current;
}

SolverOutput tabu_search(
    const Instance& instance,
    const std::optional<Routes>& initial_routes,
    const TabuOptions& options
) {
    validate_move_names(options.moves);

    std::mt19937 rng(options.seed);
    Routes current = initial_routes.has_value() ? *initial_routes : nearest_neighbor(instance);

    if (!feasible_solution(current, instance)) {
        throw std::runtime_error("Initial routes are not feasible for this instance");
    }

    double current_cost = solution_cost(current, instance);
    Routes best = current;
    double best_cost = current_cost;

    std::unordered_map<std::string, int> tabu_until;
    int no_improve = 0;

    SolverOutput out;

    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        out.iterations = iteration;

        std::optional<Routes> best_candidate;
        double best_candidate_cost = std::numeric_limits<double>::infinity();
        std::string best_candidate_sig;

        for (int sample = 0; sample < options.neighborhood_samples; ++sample) {
            const std::string& move_name = random_choice(options.moves, rng);
            auto candidate = apply_move_name(move_name, current, rng, options.lam);
            if (!candidate.has_value()) {
                continue;
            }
            if (!feasible_solution(*candidate, instance)) {
                continue;
            }

            const double cand_cost = solution_cost(*candidate, instance);
            const std::string sig = routes_signature(*candidate);
            const bool is_tabu = tabu_until.count(sig) && tabu_until[sig] > iteration;

            if (is_tabu && !(options.aspiration && cand_cost < best_cost - kEps)) {
                continue;
            }

            if (cand_cost < best_candidate_cost) {
                best_candidate = *candidate;
                best_candidate_cost = cand_cost;
                best_candidate_sig = sig;
            }
        }

        if (!best_candidate.has_value()) {
            if (options.diversification) {
                current = kick_solution(current, options.moves, rng, options.lam, options.kick_strength, instance);
                current_cost = solution_cost(current, instance);
                ++no_improve;
                out.history_best_cost.push_back(best_cost);
                continue;
            }
            break;
        }

        current = *best_candidate;
        current_cost = best_candidate_cost;

        const int extra = std::max(1, options.tabu_tenure / 4);
        const int tenure = options.tabu_tenure + rand_int(rng, 0, extra);
        tabu_until[best_candidate_sig] = iteration + tenure;

        if (current_cost < best_cost - kEps) {
            best = current;
            best_cost = current_cost;
            no_improve = 0;
        } else {
            ++no_improve;
        }

        if (options.diversification && no_improve >= options.diversification_limit) {
            current = kick_solution(current, options.moves, rng, options.lam, options.kick_strength, instance);
            current_cost = solution_cost(current, instance);
            no_improve = 0;
        }

        if (iteration % 100 == 0) {
            std::vector<std::string> expired;
            for (const auto& [sig, until] : tabu_until) {
                if (until <= iteration) {
                    expired.push_back(sig);
                }
            }
            for (const auto& sig : expired) {
                tabu_until.erase(sig);
            }
        }

        out.history_best_cost.push_back(best_cost);
    }

    out.best_routes = best;
    out.best_cost = best_cost;
    out.final_routes = current;
    out.final_cost = current_cost;

    return out;
}

SolverOutput simulated_annealing(
    const Instance& instance,
    const std::optional<Routes>& initial_routes,
    const SimulatedAnnealingOptions& options
) {
    validate_move_names(options.moves);

    std::mt19937 rng(options.seed);
    Routes current = initial_routes.has_value() ? *initial_routes : nearest_neighbor(instance);

    if (!feasible_solution(current, instance)) {
        throw std::runtime_error("Initial routes are not feasible for this instance");
    }

    double current_cost = solution_cost(current, instance);
    Routes best = current;
    double best_cost = current_cost;

    double temperature = options.initial_temperature.has_value()
        ? *options.initial_temperature
        : default_initial_temperature(current_cost);
    if (temperature <= 0.0) {
        throw std::runtime_error("initial_temperature must be > 0");
    }

    std::vector<double> move_weights(options.moves.size(), 1.0);
    int no_improve_steps = 0;

    SolverOutput out;

    while (temperature > options.final_temperature && out.iterations < options.max_iterations) {
        for (int inner = 0; inner < options.iterations_per_temperature; ++inner) {
            if (out.iterations >= options.max_iterations) {
                break;
            }
            ++out.iterations;

            std::optional<Routes> neighbor;
            int chosen_idx = -1;

            for (int attempt = 0; attempt < options.max_neighbor_attempts; ++attempt) {
                if (options.adaptive_moves) {
                    chosen_idx = roulette_index(move_weights, rng);
                } else {
                    chosen_idx = rand_int(rng, 0, static_cast<int>(options.moves.size()) - 1);
                }
                const std::string& move_name = options.moves[static_cast<std::size_t>(chosen_idx)];
                auto candidate = apply_move_name(move_name, current, rng, options.lam);
                if (!candidate.has_value()) {
                    continue;
                }
                if (!feasible_solution(*candidate, instance)) {
                    continue;
                }
                neighbor = *candidate;
                break;
            }

            if (!neighbor.has_value()) {
                continue;
            }

            const double neighbor_cost = solution_cost(*neighbor, instance);
            const double delta = neighbor_cost - current_cost;

            bool accept = false;
            if (delta <= 0.0) {
                accept = true;
            } else {
                const double prob = std::exp(-delta / std::max(temperature, 1e-12));
                if (rand01(rng) < prob) {
                    accept = true;
                }
            }

            if (accept) {
                current = *neighbor;
                current_cost = neighbor_cost;
                if (delta < 0.0) {
                    no_improve_steps = 0;
                    if (options.adaptive_moves && chosen_idx >= 0) {
                        move_weights[static_cast<std::size_t>(chosen_idx)] *= 1.03;
                    }
                } else {
                    ++no_improve_steps;
                    if (options.adaptive_moves && chosen_idx >= 0) {
                        move_weights[static_cast<std::size_t>(chosen_idx)] *= 0.999;
                    }
                }

                if (current_cost < best_cost - kEps) {
                    best = current;
                    best_cost = current_cost;
                    no_improve_steps = 0;
                }
            } else {
                ++no_improve_steps;
                if (options.adaptive_moves && chosen_idx >= 0) {
                    move_weights[static_cast<std::size_t>(chosen_idx)] *= 0.997;
                }
            }
        }

        out.history_best_cost.push_back(best_cost);

        if (options.reheat && no_improve_steps >= options.stagnation_limit) {
            temperature *= options.reheat_factor;
            no_improve_steps = 0;
        } else {
            temperature *= options.cooling_rate;
        }
    }

    out.best_routes = best;
    out.best_cost = best_cost;
    out.final_routes = current;
    out.final_cost = current_cost;
    return out;
}

SolverOutput iterated_local_search(
    const Instance& instance,
    const std::optional<Routes>& initial_routes,
    const IteratedLocalSearchOptions& options
) {
    validate_move_names(options.perturbation_moves);

    std::mt19937 rng(options.seed);

    Routes current = initial_routes.has_value() ? *initial_routes : nearest_neighbor(instance);
    if (!feasible_solution(current, instance)) {
        throw std::runtime_error("Initial routes are not feasible for this instance");
    }

    current = local_search_round(
        instance,
        current,
        options.lam,
        options.intra_methods,
        options.inter_methods,
        options.local_search_rounds
    );

    double current_cost = solution_cost(current, instance);
    Routes best = current;
    double best_cost = current_cost;

    double temp = std::max(1e-9, options.acceptance_temperature);
    int no_improve = 0;
    int strength = std::max(1, options.perturbation_strength);

    SolverOutput out;

    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        out.iterations = iteration;

        Routes perturbed = perturb_solution(
            current,
            strength,
            options.perturbation_moves,
            rng,
            options.lam,
            instance
        );

        Routes candidate = local_search_round(
            instance,
            perturbed,
            options.lam,
            options.intra_methods,
            options.inter_methods,
            options.local_search_rounds
        );

        if (!feasible_solution(candidate, instance)) {
            continue;
        }

        const double cand_cost = solution_cost(candidate, instance);
        const double delta = cand_cost - current_cost;

        bool accept = false;
        if (options.acceptance == "better") {
            accept = delta < -kEps;
        } else if (options.acceptance == "metropolis") {
            if (delta <= 0.0) {
                accept = true;
            } else {
                const double prob = std::exp(-delta / std::max(temp, 1e-12));
                accept = rand01(rng) < prob;
            }
        } else {
            throw std::runtime_error("acceptance must be 'better' or 'metropolis'");
        }

        if (accept) {
            current = candidate;
            current_cost = cand_cost;
        }

        if (cand_cost < best_cost - kEps) {
            best = candidate;
            best_cost = cand_cost;
            no_improve = 0;
            strength = std::max(1, options.perturbation_strength);
        } else {
            ++no_improve;
        }

        if (options.diversification && no_improve >= options.stagnation_limit) {
            strength = std::min(options.max_perturbation_strength, strength + 1);
            no_improve = 0;
        }

        temp *= options.cooling_rate;
        out.history_best_cost.push_back(best_cost);
    }

    out.best_routes = best;
    out.best_cost = best_cost;
    out.final_routes = current;
    out.final_cost = current_cost;
    return out;
}

SolverOutput large_neighborhood_search(
    const Instance& instance,
    const std::optional<Routes>& initial_routes,
    const LargeNeighborhoodSearchOptions& options
) {
    std::mt19937 rng(options.seed);

    Routes current = initial_routes.has_value() ? *initial_routes : nearest_neighbor(instance);
    if (!feasible_solution(current, instance)) {
        throw std::runtime_error("Initial routes are not feasible for this instance");
    }

    current = intensify_with_local_search(instance, current, options.use_local_search, options.lam);
    double current_cost = solution_cost(current, instance);

    Routes best = current;
    double best_cost = current_cost;

    std::vector<std::string> destroy_ops = {"random", "related", "worst"};
    std::vector<std::string> repair_ops = {"greedy", "regret2"};
    std::vector<double> destroy_scores = {1.0, 1.0, 1.0};
    std::vector<double> repair_scores = {1.0, 1.0};

    double temp = std::max(1e-9, options.acceptance_temperature);
    double min_destroy_fraction = options.min_destroy_fraction;
    double max_destroy_fraction = options.max_destroy_fraction;
    int no_improve = 0;

    SolverOutput out;

    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        out.iterations = iteration;

        const int n_customers = static_cast<int>(flatten_routes(current).size());
        const int min_q = std::max(1, static_cast<int>(min_destroy_fraction * n_customers));
        const int max_q = std::max(min_q, static_cast<int>(max_destroy_fraction * n_customers));
        const int q = rand_int(rng, min_q, max_q);

        const int d_idx = roulette_index(destroy_scores, rng);
        const int r_idx = roulette_index(repair_scores, rng);

        std::pair<Routes, std::vector<CustomerId>> destroyed;
        if (destroy_ops[static_cast<std::size_t>(d_idx)] == "random") {
            destroyed = destroy_random(current, q, rng);
        } else if (destroy_ops[static_cast<std::size_t>(d_idx)] == "related") {
            destroyed = destroy_related(current, q, rng, instance);
        } else {
            destroyed = destroy_worst(current, q, rng, instance);
        }

        std::optional<Routes> candidate;
        if (repair_ops[static_cast<std::size_t>(r_idx)] == "greedy") {
            candidate = repair_greedy(destroyed.first, destroyed.second, instance);
        } else {
            candidate = repair_regret2(destroyed.first, destroyed.second, instance);
        }

        if (!candidate.has_value()) {
            destroy_scores[static_cast<std::size_t>(d_idx)] *= 0.98;
            repair_scores[static_cast<std::size_t>(r_idx)] *= 0.98;
            out.history_best_cost.push_back(best_cost);
            continue;
        }

        if ((iteration % std::max(1, options.local_search_every)) == 0) {
            *candidate = intensify_with_local_search(instance, *candidate, options.use_local_search, options.lam);
        }

        if (!feasible_solution(*candidate, instance)) {
            destroy_scores[static_cast<std::size_t>(d_idx)] *= 0.97;
            repair_scores[static_cast<std::size_t>(r_idx)] *= 0.97;
            out.history_best_cost.push_back(best_cost);
            continue;
        }

        const double cand_cost = solution_cost(*candidate, instance);
        const double delta = cand_cost - current_cost;

        bool accept = false;
        if (options.acceptance == "better") {
            accept = delta < -kEps;
        } else if (options.acceptance == "metropolis") {
            if (delta <= 0.0) {
                accept = true;
            } else {
                const double prob = std::exp(-delta / std::max(temp, 1e-12));
                accept = rand01(rng) < prob;
            }
        } else {
            throw std::runtime_error("acceptance must be 'better' or 'metropolis'");
        }

        if (accept) {
            current = *candidate;
            current_cost = cand_cost;
            destroy_scores[static_cast<std::size_t>(d_idx)] *= 1.01;
            repair_scores[static_cast<std::size_t>(r_idx)] *= 1.01;
        } else {
            destroy_scores[static_cast<std::size_t>(d_idx)] *= 0.999;
            repair_scores[static_cast<std::size_t>(r_idx)] *= 0.999;
        }

        if (cand_cost < best_cost - kEps) {
            best = *candidate;
            best_cost = cand_cost;
            no_improve = 0;
            destroy_scores[static_cast<std::size_t>(d_idx)] *= 1.03;
            repair_scores[static_cast<std::size_t>(r_idx)] *= 1.03;
        } else {
            ++no_improve;
        }

        if (options.diversification && no_improve >= options.stagnation_limit) {
            min_destroy_fraction = std::min(0.45, min_destroy_fraction + 0.03);
            max_destroy_fraction = std::min(0.70, max_destroy_fraction + 0.05);
            no_improve = 0;
        } else {
            min_destroy_fraction = std::max(0.10, min_destroy_fraction * 0.999);
            max_destroy_fraction = std::max(0.30, max_destroy_fraction * 0.999);
        }

        temp *= options.cooling_rate;
        out.history_best_cost.push_back(best_cost);
    }

    out.best_routes = best;
    out.best_cost = best_cost;
    out.final_routes = current;
    out.final_cost = current_cost;
    return out;
}

SolverOutput genetic_algorithm(
    const Instance& instance,
    const std::optional<Routes>& initial_routes,
    const GeneticOptions& options
) {
    if (options.population_size < 2) {
        throw std::runtime_error("population_size must be >= 2");
    }
    if (options.elitism < 0 || options.elitism >= options.population_size) {
        throw std::runtime_error("elitism must be in [0, population_size - 1]");
    }

    std::mt19937 rng(options.seed);

    std::unordered_set<CustomerId> customer_set(instance.customer_ids.begin(), instance.customer_ids.end());
    auto population = seed_population(instance, options.population_size, rng, 3);

    if (initial_routes.has_value()) {
        auto init_perm = routes_to_permutation(*initial_routes);
        if (valid_permutation(init_perm, customer_set)) {
            population[0] = init_perm;
        }
    }

    std::vector<double> fitness;
    std::vector<Routes> decoded;
    fitness.reserve(population.size());
    decoded.reserve(population.size());

    for (const auto& perm : population) {
        auto [cost, routes] = evaluate_permutation(perm, instance);
        fitness.push_back(cost);
        decoded.push_back(routes);
    }

    int best_idx = static_cast<int>(std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end())));
    auto best_perm = population[static_cast<std::size_t>(best_idx)];
    Routes best_routes = decoded[static_cast<std::size_t>(best_idx)];
    double best_cost = fitness[static_cast<std::size_t>(best_idx)];

    int no_improve = 0;
    SolverOutput out;

    for (int generation = 1; generation <= options.generations; ++generation) {
        std::vector<int> ranking(population.size());
        std::iota(ranking.begin(), ranking.end(), 0);
        std::sort(ranking.begin(), ranking.end(), [&](int a, int b) {
            return fitness[static_cast<std::size_t>(a)] < fitness[static_cast<std::size_t>(b)];
        });

        std::vector<std::vector<CustomerId>> new_population;
        for (int i = 0; i < options.elitism; ++i) {
            new_population.push_back(population[static_cast<std::size_t>(ranking[static_cast<std::size_t>(i)])]);
        }

        while (static_cast<int>(new_population.size()) < options.population_size) {
            auto p1 = tournament_select(population, fitness, options.tournament_size, rng);
            auto p2 = tournament_select(population, fitness, options.tournament_size, rng);

            std::vector<CustomerId> child;
            if (rand01(rng) < options.crossover_rate) {
                child = order_crossover(p1, p2, rng);
            } else {
                child = p1;
            }

            if (rand01(rng) < options.mutation_rate) {
                mutate(&child, rng);
            }

            if (options.use_local_search && rand01(rng) < options.local_search_prob) {
                Routes child_routes = decode_permutation(child, instance);
                child_routes = intensify_with_local_search(instance, child_routes, true, options.lam);
                child = routes_to_permutation(child_routes);
            }

            if (!valid_permutation(child, customer_set)) {
                child = instance.customer_ids;
                std::shuffle(child.begin(), child.end(), rng);
            }

            new_population.push_back(std::move(child));
        }

        population = std::move(new_population);
        fitness.clear();
        decoded.clear();

        for (const auto& perm : population) {
            auto [cost, routes] = evaluate_permutation(perm, instance);
            fitness.push_back(cost);
            decoded.push_back(routes);
        }

        const int gen_best_idx = static_cast<int>(std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end())));
        const double gen_best_cost = fitness[static_cast<std::size_t>(gen_best_idx)];

        if (gen_best_cost < best_cost - kEps) {
            best_cost = gen_best_cost;
            best_perm = population[static_cast<std::size_t>(gen_best_idx)];
            best_routes = decoded[static_cast<std::size_t>(gen_best_idx)];
            no_improve = 0;
        } else {
            ++no_improve;
        }

        if (options.diversification && no_improve >= options.stagnation_limit) {
            std::vector<int> rank(population.size());
            std::iota(rank.begin(), rank.end(), 0);
            std::sort(rank.begin(), rank.end(), [&](int a, int b) {
                return fitness[static_cast<std::size_t>(a)] < fitness[static_cast<std::size_t>(b)];
            });

            const int restart_count = std::max(1, static_cast<int>(options.restart_fraction * options.population_size));
            for (int i = 0; i < restart_count; ++i) {
                const int idx = rank[static_cast<std::size_t>(rank.size() - 1 - i)];
                auto perm = instance.customer_ids;
                std::shuffle(perm.begin(), perm.end(), rng);
                population[static_cast<std::size_t>(idx)] = perm;
                auto [cost, routes] = evaluate_permutation(perm, instance);
                fitness[static_cast<std::size_t>(idx)] = cost;
                decoded[static_cast<std::size_t>(idx)] = routes;
            }
            no_improve = 0;
        }

        out.history_best_cost.push_back(best_cost);
        out.iterations = generation;
    }

    Routes final_best_routes = decode_permutation(best_perm, instance);

    out.best_routes = final_best_routes;
    out.best_cost = best_cost;
    out.final_routes = final_best_routes;
    out.final_cost = best_cost;
    return out;
}

SolverOutput memetic_algorithm(
    const Instance& instance,
    const std::optional<Routes>& initial_routes,
    const MemeticOptions& options
) {
    if (options.population_size < 2) {
        throw std::runtime_error("population_size must be >= 2");
    }
    if (options.elitism < 0 || options.elitism >= options.population_size) {
        throw std::runtime_error("elitism must be in [0, population_size - 1]");
    }

    std::mt19937 rng(options.seed);

    std::unordered_set<CustomerId> customer_set(instance.customer_ids.begin(), instance.customer_ids.end());
    auto population = seed_population(instance, options.population_size, rng, 4);

    if (initial_routes.has_value()) {
        auto init_perm = routes_to_permutation(*initial_routes);
        if (valid_permutation(init_perm, customer_set)) {
            population[0] = init_perm;
        }
    }

    auto educate = [&](const std::vector<CustomerId>& perm) {
        Routes routes = decode_permutation(perm, instance);

        IntraImproveOptions intra;
        intra.methods = options.intra_methods;
        intra.max_passes = options.intra_passes;

        InterImproveOptions inter;
        inter.methods = options.inter_methods;
        inter.max_passes = options.inter_passes;
        inter.lam = options.lam;

        routes = improve_intra(instance, routes, intra);
        routes = improve_inter(instance, routes, inter);
        return routes_to_permutation(routes);
    };

    std::vector<double> fitness;
    std::vector<Routes> decoded;
    fitness.reserve(population.size());
    decoded.reserve(population.size());

    for (const auto& perm : population) {
        auto [cost, routes] = evaluate_permutation(perm, instance);
        fitness.push_back(cost);
        decoded.push_back(routes);
    }

    int best_idx = static_cast<int>(std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end())));
    auto best_perm = population[static_cast<std::size_t>(best_idx)];
    double best_cost = fitness[static_cast<std::size_t>(best_idx)];

    int no_improve = 0;
    SolverOutput out;

    for (int generation = 1; generation <= options.generations; ++generation) {
        std::vector<int> ranking(population.size());
        std::iota(ranking.begin(), ranking.end(), 0);
        std::sort(ranking.begin(), ranking.end(), [&](int a, int b) {
            return fitness[static_cast<std::size_t>(a)] < fitness[static_cast<std::size_t>(b)];
        });

        std::vector<std::vector<CustomerId>> new_population;
        for (int i = 0; i < options.elitism; ++i) {
            new_population.push_back(population[static_cast<std::size_t>(ranking[static_cast<std::size_t>(i)])]);
        }

        while (static_cast<int>(new_population.size()) < options.population_size) {
            auto p1 = tournament_select(population, fitness, options.tournament_size, rng);
            auto p2 = tournament_select(population, fitness, options.tournament_size, rng);

            std::vector<CustomerId> child;
            if (rand01(rng) < options.crossover_rate) {
                child = order_crossover(p1, p2, rng);
            } else {
                child = p1;
            }

            if (rand01(rng) < options.mutation_rate) {
                mutate(&child, rng);
            }

            if (rand01(rng) < options.offspring_local_search_prob) {
                child = educate(child);
            }

            if (!valid_permutation(child, customer_set)) {
                child = instance.customer_ids;
                std::shuffle(child.begin(), child.end(), rng);
            }

            new_population.push_back(std::move(child));
        }

        population = std::move(new_population);

        if (options.elite_local_search_count > 0 && options.elite_local_search_period > 0
            && generation % options.elite_local_search_period == 0) {
            std::vector<int> rank(population.size());
            std::iota(rank.begin(), rank.end(), 0);
            std::sort(rank.begin(), rank.end(), [&](int a, int b) {
                return evaluate_permutation(population[static_cast<std::size_t>(a)], instance).first
                    < evaluate_permutation(population[static_cast<std::size_t>(b)], instance).first;
            });

            const int count = std::min(options.elite_local_search_count, static_cast<int>(population.size()));
            for (int i = 0; i < count; ++i) {
                const int idx = rank[static_cast<std::size_t>(i)];
                population[static_cast<std::size_t>(idx)] = educate(population[static_cast<std::size_t>(idx)]);
            }
        }

        fitness.clear();
        decoded.clear();
        for (const auto& perm : population) {
            auto [cost, routes] = evaluate_permutation(perm, instance);
            fitness.push_back(cost);
            decoded.push_back(routes);
        }

        const int gen_best_idx = static_cast<int>(std::distance(fitness.begin(), std::min_element(fitness.begin(), fitness.end())));
        const double gen_best_cost = fitness[static_cast<std::size_t>(gen_best_idx)];

        if (gen_best_cost < best_cost - kEps) {
            best_cost = gen_best_cost;
            best_perm = population[static_cast<std::size_t>(gen_best_idx)];
            no_improve = 0;
        } else {
            ++no_improve;
        }

        if (options.diversification && no_improve >= options.stagnation_limit) {
            std::vector<int> rank(population.size());
            std::iota(rank.begin(), rank.end(), 0);
            std::sort(rank.begin(), rank.end(), [&](int a, int b) {
                return fitness[static_cast<std::size_t>(a)] < fitness[static_cast<std::size_t>(b)];
            });

            const int restart_count = std::max(1, static_cast<int>(options.restart_fraction * options.population_size));
            for (int i = 0; i < restart_count; ++i) {
                const int idx = rank[static_cast<std::size_t>(rank.size() - 1 - i)];
                auto perm = instance.customer_ids;
                std::shuffle(perm.begin(), perm.end(), rng);
                population[static_cast<std::size_t>(idx)] = perm;
            }
            no_improve = 0;
        }

        out.history_best_cost.push_back(best_cost);
        out.iterations = generation;
    }

    Routes best_routes = decode_permutation(best_perm, instance);

    out.best_routes = best_routes;
    out.best_cost = best_cost;
    out.final_routes = best_routes;
    out.final_cost = best_cost;
    return out;
}

SolverOutput ant_colony_optimization(
    const Instance& instance,
    const std::optional<Routes>& initial_routes,
    const AcoOptions& options
) {
    if (options.num_ants < 1) {
        throw std::runtime_error("num_ants must be >= 1");
    }
    if (!(options.evaporation_rate >= 0.0 && options.evaporation_rate < 1.0)) {
        throw std::runtime_error("evaporation_rate must be in [0, 1)");
    }
    if (!(options.q0 >= 0.0 && options.q0 <= 1.0)) {
        throw std::runtime_error("q0 must be in [0, 1]");
    }
    if (options.pheromone_init <= 0.0) {
        throw std::runtime_error("pheromone_init must be > 0");
    }

    std::mt19937 rng(options.seed);

    auto dist = build_distance_matrix(instance, instance.customer_ids);
    auto eta = inverse_distance(dist);
    const int n_internal = static_cast<int>(instance.customer_ids.size()) + 1;
    std::vector<std::vector<double>> tau(static_cast<std::size_t>(n_internal),
        std::vector<double>(static_cast<std::size_t>(n_internal), options.pheromone_init));

    Routes current_seed = initial_routes.has_value() ? *initial_routes : nearest_neighbor(instance);
    if (!feasible_solution(current_seed, instance)) {
        throw std::runtime_error("Initial routes are not feasible for this instance");
    }

    Routes best = current_seed;
    double best_cost = solution_cost(best, instance);

    SolverOutput out;

    for (int iteration = 1; iteration <= options.iterations; ++iteration) {
        out.iterations = iteration;

        std::vector<Routes> ant_solutions;
        std::vector<double> ant_costs;

        for (int ant = 0; ant < options.num_ants; ++ant) {
            Routes ant_routes = construct_ant_solution(
                instance,
                instance.customer_ids,
                tau,
                eta,
                options.alpha,
                options.beta,
                options.q0,
                rng
            );

            if (options.use_local_search && rand01(rng) < options.local_search_prob) {
                ant_routes = intensify_with_local_search(instance, ant_routes, true, options.lam);
            }

            if (!feasible_solution(ant_routes, instance)) {
                continue;
            }

            const double cost = solution_cost(ant_routes, instance);
            ant_solutions.push_back(ant_routes);
            ant_costs.push_back(cost);
        }

        if (ant_solutions.empty()) {
            const double evap = 1.0 - options.evaporation_rate;
            for (auto& row : tau) {
                for (double& value : row) {
                    value *= evap;
                }
            }
            bound_pheromone(&tau, options.pheromone_min, options.pheromone_max);
            out.history_best_cost.push_back(best_cost);
            continue;
        }

        const int iter_best_idx = static_cast<int>(std::distance(ant_costs.begin(), std::min_element(ant_costs.begin(), ant_costs.end())));
        const Routes& iter_best_routes = ant_solutions[static_cast<std::size_t>(iter_best_idx)];
        const double iter_best_cost = ant_costs[static_cast<std::size_t>(iter_best_idx)];

        if (iter_best_cost < best_cost - kEps) {
            best_cost = iter_best_cost;
            best = iter_best_routes;
        }

        const double evap = 1.0 - options.evaporation_rate;
        for (auto& row : tau) {
            for (double& value : row) {
                value *= evap;
            }
        }

        const double delta_iter = 1.0 / std::max(iter_best_cost, 1e-12);
        deposit_pheromone(iter_best_routes, delta_iter, &tau, instance.customer_ids);

        const double delta_global = options.elite_weight / std::max(best_cost, 1e-12);
        deposit_pheromone(best, delta_global, &tau, instance.customer_ids);

        bound_pheromone(&tau, options.pheromone_min, options.pheromone_max);

        out.history_best_cost.push_back(best_cost);
    }

    out.best_routes = best;
    out.best_cost = best_cost;
    out.final_routes = best;
    out.final_cost = best_cost;
    return out;
}

SolverOutput output_from_routes(const Instance& instance, const Routes& routes) {
    SolverOutput out;
    out.best_routes = routes;
    out.best_cost = solution_cost(routes, instance);
    out.final_routes = routes;
    out.final_cost = out.best_cost;
    out.iterations = 1;
    return out;
}

std::string routes_to_json(const Routes& routes) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < routes.size(); ++i) {
        if (i) {
            oss << ',';
        }
        oss << '[';
        for (std::size_t j = 0; j < routes[i].size(); ++j) {
            if (j) {
                oss << ',';
            }
            oss << routes[i][j];
        }
        oss << ']';
    }
    oss << ']';
    return oss.str();
}

std::string solver_output_to_json(const SolverOutput& output) {
    std::ostringstream oss;
    oss << std::setprecision(15);
    oss << "{";
    oss << "\"status\":\"ok\",";
    oss << "\"best_cost\":" << output.best_cost << ',';
    oss << "\"best_routes\":" << routes_to_json(output.best_routes) << ',';
    oss << "\"final_cost\":" << output.final_cost << ',';
    oss << "\"final_routes\":" << routes_to_json(output.final_routes) << ',';
    oss << "\"iterations\":" << output.iterations;
    if (!output.history_best_cost.empty()) {
        oss << ",\"history_best_cost\":[";
        for (std::size_t i = 0; i < output.history_best_cost.size(); ++i) {
            if (i) {
                oss << ',';
            }
            oss << output.history_best_cost[i];
        }
        oss << ']';
    }
    oss << '}';
    return oss.str();
}

std::string error_to_json(const std::string& message) {
    std::ostringstream oss;
    oss << "{\"status\":\"error\",\"error\":\"" << json_escape(message) << "\"}";
    return oss.str();
}

std::unordered_map<std::string, std::string> parse_cli_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (!starts_with(arg, "--")) {
            continue;
        }

        arg = arg.substr(2);

        const auto eq = arg.find('=');
        if (eq != std::string::npos) {
            const std::string key = arg.substr(0, eq);
            const std::string value = arg.substr(eq + 1);
            args[key] = value;
            continue;
        }

        const std::string key = arg;
        if (i + 1 < argc) {
            std::string next = argv[i + 1];
            if (!starts_with(next, "--")) {
                args[key] = next;
                ++i;
                continue;
            }
        }

        args[key] = "true";
    }

    return args;
}

std::vector<std::string> split_csv(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

bool as_bool(const std::string& value) {
    const std::string v = to_lower(trim(value));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

}  // namespace cvrp
