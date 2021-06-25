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

add_executable(example_hypre hypre_example_solve.cpp)

target_link_libraries(example_hypre PUBLIC
        Alien::alien_hypre
        Alien::alien_semantic_ref)

enable_testing()
include(LoadAlienTest)

alien_test(
    BENCH hypre
    NAME hypre.example
    PROCS 4
    COMMAND example_hypre
)

#-----------------------------------------------------------
# Bench recap
#-----------------------------------------------------------

print_bench_informations()

