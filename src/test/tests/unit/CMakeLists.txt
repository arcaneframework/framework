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

# tests with GTest

set(SEQ_TEST_LIST
        Space
        AlienInclude
        )

set(PARALLEL_TEST_LIST
        BackEnds
        Timestamp
        CompositeSpace
        Universe
        Distribution
        SpacePartition
        DoKMatrix
        Redistributor
        )

add_executable(gtest.unitary main.cpp)

foreach (test_name ${SEQ_TEST_LIST})
    target_sources(gtest.unitary PRIVATE Test${test_name}.cc)
endforeach ()

foreach (test_name ${PARALLEL_TEST_LIST})
    target_sources(gtest.unitary PRIVATE Test${test_name}.cc)
endforeach ()

target_link_libraries(gtest.unitary
        PRIVATE GTest::GTest
        PRIVATE Alien::alien_core
        alien_test_framework
        )

foreach (test_name ${SEQ_TEST_LIST})
    alien_test(
            BENCH gtest
            NAME ${test_name}
            COMMAND gtest.unitary
            OPTIONS --gtest_filter=Test${test_name}*
    )
endforeach ()

foreach (test_name ${PARALLEL_TEST_LIST})
    alien_test(
            BENCH gtest
            NAME ${test_name}
            PROCS 2
            COMMAND gtest.unitary
            OPTIONS --gtest_filter=Test${test_name}*
    )
endforeach ()
