# Copyright 2022 IFPEN-CEA
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

project(AlienTrilinosExample VERSION 1.0 LANGUAGES CXX)

find_package(Alien REQUIRED)

add_executable(example_trilinos trilinos_example_solve.cpp)

target_link_libraries(example_trilinos PUBLIC
        Alien::trilinos_wrapper
        Alien::alien_semantic_move)

target_include_directories(trilinos_wrapper PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

configure_file(msc00726.mtx msc00726.mtx COPYONLY)
configure_file(mesh1em6.mtx mesh1em6.mtx COPYONLY)

if (ALIEN_BENCHMARK)
    enable_testing()
    include(LoadAlienTest)

    alien_test(
            BENCH trilinos_wrapper
            NAME trilinos.example.cg
            PROCS 2
            COMMAND example_trilinos
            OPTIONS CG Relaxation msc00726.mtx
    )

    alien_test(
            BENCH trilinos_wrapper
            NAME trilinos.example.gmres
            PROCS 2
            COMMAND example_trilinos
            OPTIONS GMRES Relaxation mesh1em6.mtx
    )
    #-----------------------------------------------------------
    # Bench recap
    #-----------------------------------------------------------
    print_bench_informations()
endif (ALIEN_BENCHMARK)


