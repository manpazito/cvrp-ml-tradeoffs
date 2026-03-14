cmake_minimum_required(VERSION 3.16)

project(cvrp_models LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cvrp_core src/cvrp_core.cpp)
target_include_directories(cvrp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(cvrp_constructive src/constructive_main.cpp)
add_executable(cvrp_improve_intra src/improve_intra_main.cpp)
add_executable(cvrp_improve_inter src/improve_inter_main.cpp)
add_executable(cvrp_tabu src/tabu_main.cpp)
add_executable(cvrp_simulated_annealing src/simulated_annealing_main.cpp)
add_executable(cvrp_iterated_local_search src/iterated_local_search_main.cpp)
add_executable(cvrp_large_neighborhood_search src/large_neighborhood_search_main.cpp)
add_executable(cvrp_genetic src/genetic_main.cpp)
add_executable(cvrp_memetic src/memetic_main.cpp)
add_executable(cvrp_ant_colony src/ant_colony_main.cpp)

foreach(target
    cvrp_constructive
    cvrp_improve_intra
    cvrp_improve_inter
    cvrp_tabu
    cvrp_simulated_annealing
    cvrp_iterated_local_search
    cvrp_large_neighborhood_search
    cvrp_genetic
    cvrp_memetic
    cvrp_ant_colony)
    target_link_libraries(${target} PRIVATE cvrp_core)
endforeach()

