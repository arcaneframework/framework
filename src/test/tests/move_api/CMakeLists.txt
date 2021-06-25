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

cmake_minimum_required(VERSION 3.0)

set(TEST_LIST
        Vector
        VectorBuilder
        MatrixDirectBuilder
        )

set(PARALLEL_TEST_LIST
        IndexManager)


add_executable(gtest.movesemantic
        main.cpp)

foreach (test_name ${TEST_LIST})
    target_sources(gtest.movesemantic PRIVATE Test${test_name}.cc)
endforeach ()

foreach (test_name ${PARALLEL_TEST_LIST})
    target_sources(gtest.movesemantic PRIVATE Test${test_name}.cc)
endforeach ()

target_link_libraries(gtest.movesemantic PRIVATE
        GTest::GTest
        Alien::alien_core
        Alien::alien_semantic_move
        alien_test_framework
        )

install(
        TARGETS gtest.movesemantic
        RUNTIME DESTINATION bin/movesemantic/unitary
)

foreach (test_name ${TEST_LIST})
    alien_test(
            BENCH movesemantic.gtest
            NAME ${test_name}
            COMMAND gtest.movesemantic
            OPTIONS --gtest_filter=Test${test_name}*
    )
endforeach ()


foreach (test_name ${PARALLEL_TEST_LIST})
    alien_test(
            BENCH movesemantic.gtest
            NAME ${test_name}
            PROCS 4
            COMMAND gtest.movesemantic
            OPTIONS --gtest_filter=Test${test_name}*
    )
endforeach ()


