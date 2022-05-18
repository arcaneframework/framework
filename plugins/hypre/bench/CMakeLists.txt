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

find_package(Alien REQUIRED)


add_executable(bench_hypre bench.cpp)

target_link_libraries(bench_hypre PUBLIC
        Alien::hypre_wrapper
        Alien::alien_semantic_move
        Alien::benchmark)

include(GNUInstallDirs)

install(TARGETS bench_hypre EXPORT Alien
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

if (ALIEN_UNIT_TESTS)

    configure_file(../../../src/movesemantic/tests/simple.mtx matrix.mtx COPYONLY)
    configure_file(../../../src/movesemantic/tests/simple_rhs.mtx rhs.mtx COPYONLY)

    #-----------------------------------------------------------
    # Tests
    #-----------------------------------------------------------

    alien_test(
            BENCH hypre
            NAME hypre.benchmark
            PROCS 2
            COMMAND bench_hypre
    )
endif ()
