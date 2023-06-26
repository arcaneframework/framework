# Copyright 2020 IFPEN-CEA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.13)

# for find_package with Foo_ROOT
cmake_policy(SET CMP0074 NEW)

project(AlienHypreExample VERSION 1.0 LANGUAGES CXX)

find_package(Alien REQUIRED)

add_executable(example_hypre hypre_example_solve.cpp)

target_link_libraries(example_hypre PUBLIC
        Alien::hypre_wrapper
        Alien::alien_semantic_ref)

add_executable(example_hypre_move hypre_example_solve_move.cpp)

target_link_libraries(example_hypre_move PUBLIC
        Alien::hypre_wrapper
        Alien::alien_semantic_move)

add_executable(mm_driver_hypre driver_mm_hypre.cc)

target_link_libraries(mm_driver_hypre PUBLIC
        Alien::hypre_wrapper
        Alien::alien_semantic_move)

configure_file(msc00726.mtx msc00726.mtx COPYONLY)
configure_file(mesh1em6.mtx mesh1em6.mtx COPYONLY)

if (ALIEN_BENCHMARK)
    enable_testing()
    include(LoadAlienTest)

    alien_test(
            BENCH hypre
            NAME hypre.example
            PROCS 4
            COMMAND example_hypre
    )

    alien_test(
            BENCH hypre
            NAME hypre.example.move
            PROCS 4
            COMMAND example_hypre_move
    )

    alien_test(
            BENCH hypre
            NAME hypre.matrixmarket
            PROCS 4
            COMMAND mm_driver_hypre
    )
    #-----------------------------------------------------------
    # Bench recap
    #-----------------------------------------------------------

    print_bench_informations()
endif (ALIEN_BENCHMARK)
