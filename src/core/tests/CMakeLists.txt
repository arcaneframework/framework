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

include(LoadAlienTest)

find_package(MPI REQUIRED)
find_package(GTest REQUIRED)

# tests with GTest

add_executable(core.gtest.seq main.cpp TestFamilies.cc
        TestSpace.cc TestAlienInclude.cc TestSpacePartition.cc)

add_executable(core.gtest.mpi main.cpp
        TestFamilies.cc
        TestBackEnds.cc
        TestTimestamp.cc
        TestCompositeSpace.cc
        TestUniverse.cc
        TestDistribution.cc
        TestDoK.cc
        TestRedistributor.cc
        TestIndexManager.cc
        TestVBlockBuilder.cc
        )

target_link_libraries(core.gtest.seq
        PUBLIC GTest::GTest
        PUBLIC Alien::alien_core
        alien_test_framework
        arcconpkg_MPI
        )

target_link_libraries(core.gtest.mpi
        PUBLIC GTest::GTest
        PUBLIC Alien::alien_core
        alien_test_framework
        arcconpkg_MPI
        )

if (ALIEN_EXPAND_TESTS)
    gtest_discover_tests(core.gtest.seq TEST_PREFIX "core." TEST_SUFFIX ".seq" TEST_LIST core-seq DISCOVERY_TIMEOUT 10)
    set_tests_properties(${core-seq} PROPERTIES TIMEOUT 30)
    gtest_discover_tests(core.gtest.mpi TEST_PREFIX "core." TEST_SUFFIX ".mpi" TEST_LIST core-mpi DISCOVERY_TIMEOUT 10)
    set_tests_properties(${core-mpi} PROPERTIES TIMEOUT 30)
else (ALIEN_EXPAND_TESTS)
    alien_test(
            BENCH core
            NAME "seq"
            COMMAND core.gtest.seq
    )
    alien_test(
            BENCH core
            NAME "mpi"
            COMMAND core.gtest.mpi
    )
endif (ALIEN_EXPAND_TESTS)

alien_test(
        BENCH core
        NAME "mpi"
        PROCS 2
        COMMAND core.gtest.mpi
        PARALLEL_ONLY
)
