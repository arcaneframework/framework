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

add_library(alien_utils OBJECT
        ArrayUtils.h
        CountStampMng.h
        ICopyOnWriteObject.h
        MoveObject.h
        ObjectWithLock.h
        parameter_manager/BaseParameterManager.cc
        parameter_manager/BaseParameterManager.h
        Precomp.h
        time_stamp/ITimestampObserver.h
        time_stamp/Timestamp.cc
        time_stamp/Timestamp.h
        time_stamp/TimestampMng.cc
        time_stamp/TimestampMng.h
        time_stamp/TimestampObserver.cc
        time_stamp/TimestampObserver.h
        Trace.h
        ObjectWithTrace.h
        TupleRandomIterator.h
        UserFeatureMng.h
        VMap.h
        SafeConstArrayView.h)

target_link_libraries(alien_utils PUBLIC
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_utils PUBLIC alien_headers)

install(TARGETS alien_utils EXPORT ${ALIEN_EXPORT_TARGET})
