# Copyright 2021 IFPEN-CEA
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

# To properly handle VERSION directive in PROJECT
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0023 NEW)

# To export dependencies
cmake_policy(SET CMP0022 NEW)

# for find_package with Foo_ROOT
cmake_policy(SET CMP0074 NEW)

project(AlienTutorials VERSION 1.0 LANGUAGES CXX)

find_package(Alien REQUIRED)
find_package(Arccore REQUIRED)

add_executable(tuto_dist_space.exe dist_space.cpp)
target_link_libraries(tuto_dist_space.exe
        PRIVATE Alien::alien_core
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        )

add_executable(tuto_dist_matrix.exe dist_matrix.cpp)

target_link_libraries(tuto_dist_matrix.exe PRIVATE
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        Alien::alien_semantic_ref
        )

add_executable(tuto_dist_vector.exe dist_vector.cpp)

target_link_libraries(tuto_dist_vector.exe PRIVATE
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        Alien::alien_semantic_ref
        )

add_executable(tuto_dist_linear_system.exe dist_linear_system.cpp)

target_link_libraries(tuto_dist_linear_system.exe PRIVATE
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        Alien::alien_semantic_ref
        )
