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

add_executable(test_petsc test_petsc.cpp)

target_link_libraries(test_petsc PUBLIC
        Alien::petsc_wrapper
        Alien::alien_semantic_ref)

include(LoadAlienTest)

#-----------------------------------------------------------
# Tests
#-----------------------------------------------------------

alien_test(
        BENCH petsc
        NAME petsc.default
        PROCS 2
        COMMAND test_petsc
)

#~ #-----------------------------------------------------------
#~ # Bench recap
#~ #-----------------------------------------------------------

print_bench_informations()

