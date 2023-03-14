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

add_library(alien_test_framework
        Environment.cpp Environment.h
        CreateLinearSystemFiles.cpp CreateLinearSystemFiles.h
        )

target_link_libraries(alien_test_framework PUBLIC
        Arccore::arccore_message_passing_mpi
        Arccore::arccore_trace
        Alien::alien_core
        )

generate_export_header(alien_test_framework
        EXPORT_FILE_NAME alien/AlienTestExport.h)

target_include_directories(alien_test_framework PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        .)
