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


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake)

IF(ALIEN_USE_HIPSYCL)
  #find_package(hipSYCL CONFIG REQUIRED)
  message(status "hipSYCL FOUND          : ${IntelDPCPP_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
ENDIF()
IF(ALIEN_USE_INTELDPCPP)
  #find_package(IntelDPCPP CONFIG REQUIRED)
  message(status "IntelDPCPP             : ${IntelDPCPP_FOUND}")
  message(status "IntelDPCPP FOUND       : ${IntelDPCPP_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
ENDIF()
IF(ALIEN_USE_INTELSYCL)
  set(CMAKE_CXX_COMPILER ${ONEAPI_CXX_COMPILER})
  #find_package(IntelSYCL REQUIRED)
  message(status "IntelSYCL              : ${IntelSYCL_FOUND}")
  message(status "IntelSYCL FOUND        : ${IntelSYCL_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
  message(status "      SYCL LINK FLAGS  : ${SYCL_LINK_FLAGS}")
ENDIF()
IF(ALIEN_USE_ACPPSYCL)
  #find_package(AdaptiveCpp REQUIRED)
  message(status "AdaptiveSYCL           : ${AdaptiveCpp_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
  message(status "      SYCL LINK FLAGS  : ${SYCL_LINK_FLAGS}")
ENDIF()

find_package(Boost COMPONENTS context)

add_library(alien_handlers_sycl OBJECT
        VectorAccessorT.h
        VectorAccessorImplT.h
        ProfiledMatrixBuilderT.h
        ProfiledMatrixBuilderImplT.h
        VectorAccessor.cc
        )

IF(ALIEN_USE_HIPSYCL)
  IF(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(alien_kernel_sycl PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
  ENDIF()
  set(DPCPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
  target_compile_options(alien_kernel_sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
  target_compile_definitions(alien_kernel_sycl PRIVATE USE_HIPSYCL)
  #target_include_directories(alien_kernel_sycl PRIVATE ${SYCL_INCLUDE_DIR_HINT})
  add_sycl_to_target(TARGET alien_handlers_sycl 
                     SOURCES VectorAccessor.cc)
ENDIF()
IF(ALIEN_USE_INTELSYCL)
   IF(ALIEN_USE_CUDA)
     set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
     target_compile_options(alien_handlers_sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
     target_link_options(alien_handlers_sycl PRIVATE ${DPCPP_FLAGS})
   ENDIF()
   IF(ALIEN_USE_HIP)
     set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
     #set(DPCPP_FLAGS -fsycl --amdgpu-target=gfx90a --offload-arch=gfx90a -Wno-linker-warnings)
     target_compile_options(alien_handlers_sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
     target_link_options(alien_handlers_sycl PRIVATE ${DPCPP_FLAGS})
   ENDIF()  
  add_sycl_to_target(TARGET alien_handlers_sycl 
                     SOURCES VectorAccessor.cc)
ENDIF()

IF(ALIEN_USE_ACPPSYCL)
  IF(CMAKE_COMPILER_IS_GNUCXX)
      target_compile_options(alien_handlers_sycl PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
  ENDIF()
  IF(ALIEN_USE_CUDA)
    #set(ACPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings -DUSE_SYCL2020)
    #target_compile_options(alien_handlers_sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
    #target_link_options(alien_handlers_sycl PRIVATE ${ACPP_FLAGS})
  ENDIF()
  IF(ALIEN_USE_HIP)
    set(ACPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
    target_link_options(alien_handlers_sycl PRIVATE ${ACPP_FLAGS})
  ENDIF()
  target_compile_definitions(alien_handlers_sycl PRIVATE USE_ACPPSYCL)
  add_sycl_to_target(TARGET alien_handlers_sycl 
                     SOURCES VectorAccessor.cc)
ENDIF()
target_link_libraries(alien_handlers_sycl PUBLIC
                      Arccore::arccore_trace
                      Arccore::arccore_collections
                      Arccore::arccore_base
                      Arccore::arccore_message_passing_mpi)

target_link_libraries(alien_handlers_sycl PUBLIC alien_utils alien_headers)

install(TARGETS alien_handlers_sycl EXPORT ${ALIEN_EXPORT_TARGET})


