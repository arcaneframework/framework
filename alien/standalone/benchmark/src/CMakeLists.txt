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

add_library(alien_benchmark
        alien/benchmark/MatrixMarketProblem.cpp
        alien/benchmark/MatrixMarketProblem.h
        alien/benchmark/ILinearProblem.h alien/benchmark/LinearBench.cpp alien/benchmark/LinearBench.h alien/benchmark/VectorAnalytics.cpp alien/benchmark/VectorAnalytics.h)

target_include_directories(alien_benchmark PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(alien_benchmark PUBLIC
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        Alien::alien_semantic_move
        Alien::alien_semantic_ref
        )

add_library(Alien::benchmark ALIAS alien_benchmark)

include(GenerateExportHeader)

generate_export_header(alien_benchmark
        BASE_NAME ALIEN_BENCHMARK
        EXPORT_FILE_NAME alien/benchmark/export.h
        )

install(TARGETS alien_benchmark EXPORT Alien
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/alien/benchmark/export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/benchmark
)