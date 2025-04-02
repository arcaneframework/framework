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

IF(ALIEN_USE_HIPSYCL)
  find_package(hipSYCL CONFIG REQUIRED)
  message(status "hipSYCL FOUND          : ${IntelDPCPP_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
ENDIF()

IF(ALIEN_USE_INTELDPCPP)
  find_package(IntelDPCPP CONFIG REQUIRED)
  message(status "IntelDPCPP             : ${IntelDPCPP_FOUND}")
  message(status "IntelDPCPP FOUND       : ${IntelDPCPP_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
ENDIF()
IF(ALIEN_USE_INTELSYCL)
  set(CMAKE_CXX_COMPILER ${ONEAPI_CXX_COMPILER})
  find_package(IntelSYCL REQUIRED)
  message(status "IntelSYCL              : ${IntelSYCL_FOUND}")
  message(status "IntelSYCL FOUND        : ${IntelSYCL_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
  message(status "      SYCL LINK FLAGS  : ${SYCL_LINK_FLAGS}")
ENDIF()

IF(ALIEN_USE_ACPPSYCL)
    find_package(AdaptiveCpp REQUIRED)
    message(status "AdaptiveSYCL           : ${AdaptiveCpp_FOUND}")
    message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
    message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
    message(status "      SYCL LINK FLAGS  : ${SYCL_LINK_FLAGS}")
    find_package(Boost COMPONENTS fiber context REQUIRED)
ENDIF()

add_executable(ref.gtest.mpi
        main.cpp
        TestIndexManager.cc
        TestVBlockMatrixBuilder.cc
        )

if (ALIEN_USE_LIBXML2)
    target_sources(ref.gtest.mpi PRIVATE TestImportExport.cc)
endif ()


if (ALIEN_USE_EIGEN3)
    target_sources(ref.gtest.mpi PRIVATE TestSchur.cc)
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
        TestMVExpr.cc)

if (ALIEN_USE_EIGEN3)
    target_sources(ref.gtest.seq PRIVATE TestSchur.cc)
endif ()

IF (ALIEN_USE_SYCL)
    target_sources(ref.gtest.seq PRIVATE TestMVSYCL.cc)
    
    IF(ALIEN_USE_HIPSYCL)
      IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(ref.gtest.seq PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      ENDIF()
      set(DPCPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
      target_compile_options(ref.gtest.seq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
      target_compile_definitions(ref.gtest.seq PRIVATE USE_HIPSYCL)
      add_sycl_to_target(TARGET ref.gtest.seq 
                         SOURCES TestMVSYCL.cc)
    ENDIF()
    IF(ALIEN_USE_INTELSYCL)
       #set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
       IF(ALIEN_USE_CUDA)
         set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
         target_compile_options(ref.gtest.seq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
         target_link_options(ref.gtest.seq PRIVATE ${DPCPP_FLAGS})
       ENDIF()
       add_sycl_to_target(TARGET ref.gtest.seq  
                          SOURCES TestMVSYCL.cc)
    ENDIF()
    
    IF(ALIEN_USE_ACPPSYCL)
      IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(ref.gtest.seq PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      ENDIF()
      IF(ALIEN_USE_HIP)
        set(ACPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
        target_compile_options(ref.gtest.seq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
      ENDIF()
       IF(ALIEN_USE_CUDA)
         #set(ACPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
         #target_compile_options(ref.gtest.seq PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ACPP_FLAGS}>)
         #target_link_options(ref.gtest.seq PRIVATE ${ACPP_FLAGS})
       ENDIF()
       target_compile_definitions(ref.gtest.seq PRIVATE USE_ACPPSYCL)
       add_sycl_to_target(TARGET ref.gtest.seq  
                          SOURCES TestMVSYCL.cc)
    ENDIF()
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
    gtest_discover_tests(ref.gtest.seq TEST_PREFIX "alien.ref." TEST_SUFFIX ".seq" DISCOVERY_TIMEOUT 10)
    gtest_discover_tests(ref.gtest.mpi TEST_PREFIX "alien.ref." TEST_SUFFIX ".seq" DISCOVERY_TIMEOUT 10)
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


