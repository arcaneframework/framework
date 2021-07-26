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


add_executable(tuto_dist_space.exe dist_space.cpp)
target_link_libraries(tuto_dist_space.exe
        PRIVATE Alien::alien_core
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        )

install(
        TARGETS tuto_dist_space.exe
        RUNTIME DESTINATION bin
)


add_executable(tuto_dist_matrix.exe dist_matrix.cpp)

target_link_libraries(tuto_dist_matrix.exe PRIVATE
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        Alien::alien_semantic_ref
        )

install(
        TARGETS tuto_dist_matrix.exe
        RUNTIME DESTINATION
)


add_executable(tuto_dist_vector.exe dist_vector.cpp)

target_link_libraries(tuto_dist_vector.exe PRIVATE
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        Alien::alien_semantic_ref
        )

install(
        TARGETS tuto_dist_vector.exe
        RUNTIME DESTINATION
)

add_executable(tuto_dist_linear_system.exe dist_linear_system.cpp)

target_link_libraries(tuto_dist_linear_system.exe PRIVATE
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        Alien::alien_semantic_ref
        )

install(
        TARGETS tuto_dist_linear_system.exe
        RUNTIME DESTINATION
)
