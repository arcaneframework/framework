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

find_package(MPI REQUIRED)
find_package(GTest REQUIRED)

add_executable(ref.gtest.mpi
        main.cpp
        TestIndexManager.cc
        TestVBlockMatrixBuilder.cc
        )

if (ALIEN_USE_LIBXML2)
    target_sources(ref.gtest.mpi PRIVATE TestImportExport.cc)
endif ()

add_executable(ref.gtest.seq main.cpp
        TestVector.cc
        TestVectorBuilder.cc
        TestMatrixDirectBuilder.cc
        TestVBlockMatrixBuilder.cc
        TestCompositeVector.cc
        TestCompositeMatrix.cc
        TestBlockMatrix.cc
        TestVBlockMatrix.cc
        TestRedistributor.cc
        TestMVExpr.cc
        TestSchur.cc)

IF (ALIEN_USE_SYCL)
    target_sources(ref.gtest.seq PRIVATE TestMVSYCL.cc)
ENDIF ()

target_link_libraries(ref.gtest.mpi PRIVATE
        GTest::GTest
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_test_framework
        arcconpkg_MPI
        )

target_link_libraries(ref.gtest.seq PRIVATE
        GTest::GTest
        Alien::alien_core
        Alien::alien_semantic_ref
        alien_test_framework
        )

if (ALIEN_EXPAND_TESTS)
    gtest_discover_tests(ref.gtest.seq TEST_PREFIX "ref." TEST_SUFFIX ".seq" DISCOVERY_TIMEOUT 10)
    gtest_discover_tests(ref.gtest.mpi TEST_PREFIX "ref." TEST_SUFFIX ".seq" DISCOVERY_TIMEOUT 10)
else (ALIEN_EXPAND_TESTS)
    alien_test(
            BENCH ref
            NAME "seq"
            COMMAND ref.gtest.seq
    )
    alien_test(
            BENCH ref
            NAME "mpi"
            COMMAND ref.gtest.mpi
    )
endif (ALIEN_EXPAND_TESTS)

alien_test(
        BENCH ref
        NAME "mpi"
        PROCS 4
        COMMAND ref.gtest.mpi
        PARALLEL_ONLY
)


