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

add_library(alien_index_manager OBJECT
        IAbstractFamily.h
        IndexManager.h
        IndexManager.cc
        ScalarIndexSet.h
        ScalarIndexSet.cc
        functional/DefaultAbstractFamily.h
        functional/DefaultAbstractFamily.cc
        functional/AbstractItemFamily.h
        functional/AbstractItemFamily.cc
        functional/DefaultIndexManager.h
        functional/DefaultIndexManager.cc
        functional/BasicIndexManager.h
        functional/BasicIndexManager.cc
        )

target_link_libraries(alien_index_manager PUBLIC
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_index_manager PUBLIC alien_utils alien_headers)

install(TARGETS alien_index_manager EXPORT ${ALIEN_EXPORT_TARGET})
