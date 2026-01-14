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
find_package(Boost COMPONENTS program_options)

if(ALIEN_USE_SYCL)
    IF(ALIEN_USE_HIPSYCL)
      find_package(hipSYCL CONFIG REQUIRED)
      message(status "HIPSYCL                : ${hipSYCL_FOUND}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
    ENDIF()
    IF(ALIEN_USE_INTELDPCPP)
      find_package(IntelDPCPP CONFIG REQUIRED)
      message(status "IntelDPCPP : ${IntelDPCPP_FOUND}")
      message(status "IntelDPCPP FOUND : ${IntelDPCPP_FOUND}")
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

    add_executable(test_sycl.exe test_sycl.cpp)
    add_executable(test_mvhandler_sycl.exe test_mvhandler_sycl.cpp)
    add_executable(test_combine_mvhandler_sycl.exe test_combine_mvhandler_sycl.cpp)
    target_compile_definitions(test_sycl.exe PUBLIC ALIEN_USE_SYCL USE_SYCL2020)
    target_compile_definitions(test_mvhandler_sycl.exe PUBLIC ALIEN_USE_SYCL USE_SYCL2020)
    target_compile_definitions(test_combine_mvhandler_sycl.exe PUBLIC ALIEN_USE_SYCL USE_SYCL2020)
    IF(ALIEN_USE_HIPSYCL)
      add_sycl_to_target(TARGET  test_sycl.exe
                         SOURCES test_sycl.cpp)
      add_sycl_to_target(TARGET  test_mvhandler_sycl.exe
                         SOURCES test_mvhandler_sycl.cpp)
      add_sycl_to_target(TARGET  test_combine_mvhandler_sycl.exe
                         SOURCES test_combine_mvhandler_sycl.cpp)
    ENDIF()
    IF(ALIEN_USE_INTELSYCL)
       IF(ALIEN_USE_CUDA)
         set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
         target_compile_options(test_sycl.exe PRIVATE ${DPCPP_FLAGS})
         target_link_options(test_sycl.exe PRIVATE ${DPCPP_FLAGS})
         target_compile_options(test_mvhandler_sycl.exe PRIVATE ${DPCPP_FLAGS})
         target_link_options(test_mvhandler_sycl.exe PRIVATE ${DPCPP_FLAGS})
         target_compile_options(test_combine_mvhandler_sycl.exe PRIVATE ${DPCPP_FLAGS})
         target_link_options(test_combine_mvhandler_sycl.exe PRIVATE ${DPCPP_FLAGS})
       ENDIF()
       #set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
       #target_compile_options(test_sycl.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda -DUSE_SYCL2020>)
       add_sycl_to_target(TARGET  test_sycl.exe
                          SOURCES test_sycl.cpp)
       add_sycl_to_target(TARGET  test_mvhandler_sycl.exe
                          SOURCES test_mvhandler_sycl.cpp)
       add_sycl_to_target(TARGET  test_combine_mvhandler_sycl.exe
                          SOURCES test_combine_mvhandler_sycl.cpp)
    ENDIF()
    IF(ALIEN_USE_ACPPSYCL)
      IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(test_sycl.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      ENDIF()
      target_compile_definitions(test_sycl.exe PRIVATE USE_ACPPSYCL)
      add_sycl_to_target(TARGET  test_sycl.exe
                         SOURCES test_sycl.cpp)

      IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(test_mvhandler_sycl.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      ENDIF()
      target_compile_definitions(test_mvhandler_sycl.exe PRIVATE USE_ACPPSYCL)
      add_sycl_to_target(TARGET  test_mvhandler_sycl.exe
                         SOURCES test_mvhandler_sycl.cpp)

      IF(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(test_combine_mvhandler_sycl.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
      ENDIF()
      target_compile_definitions(test_combine_mvhandler_sycl.exe PRIVATE USE_ACPPSYCL)
      add_sycl_to_target(TARGET  test_combine_mvhandler_sycl.exe
                         SOURCES test_combine_mvhandler_sycl.cpp)
    ENDIF()

    target_link_libraries(test_sycl.exe PUBLIC
                      Alien::alien_core
                      Alien::alien_semantic_ref
                      arcconpkg_MPI
                      ${Boost_LIBRARIES})

    target_link_libraries(test_mvhandler_sycl.exe PUBLIC
                      Alien::alien_core
                      Alien::alien_semantic_ref
                      arcconpkg_MPI
                      ${Boost_LIBRARIES})

    target_link_libraries(test_combine_mvhandler_sycl.exe PUBLIC
                      Alien::alien_core
                      Alien::alien_semantic_ref
                      arcconpkg_MPI
                      ${Boost_LIBRARIES})

endif()


message(STATUS "Using AVX Simd instructions ? -> ${ALIEN_WANT_AVX}")
message(STATUS "Using AVX2 Simd instructions ? -> ${ALIEN_WANT_AVX2}")
message(STATUS "Using AVX512 Simd instructions ? -> ${ALIEN_WANT_AVX512}")

add_executable(krylov_example.exe test_krylov.cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID STREQUAL Clang))
  if(ALIEN_WANT_AVX)
    target_compile_options(krylov_example.exe INTERFACE -mavx)
  endif()
  if(ALIEN_WANT_AVX2)
    target_compile_options(krylov_example.exe INTERFACE -mavx -mfma)
  endif()
  if(ALIEN_WANT_AVX512)
    target_compile_options(krylov_example.exe INTERFACE -mavx512f -mavx512cd)
  endif()
endif()

if(ALIEN_USE_SYCL)
  target_compile_definitions(krylov_example.exe PUBLIC ALIEN_USE_SYCL USE_SYCL2020)
  IF(ALIEN_USE_HIPSYCL)
    add_sycl_to_target(TARGET  krylov_example.exe
                        SOURCES test_krylov.cpp)
  ENDIF()
  IF(ALIEN_USE_INTELSYCL)
    #set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
    IF(ALIEN_USE_CUDA)
      set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
      target_compile_options(krylov_example.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
    ENDIF()
    add_sycl_to_target(TARGET  krylov_example.exe
                        SOURCES test_krylov.cpp)
  ENDIF()
  IF(ALIEN_USE_ACPPSYCL)
    IF(CMAKE_COMPILER_IS_GNUCXX)
      target_compile_options(krylov_example.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
    ENDIF()
    target_compile_definitions(krylov_example.exe PRIVATE USE_ACPPSYCL)
    add_sycl_to_target(TARGET  krylov_example.exe
                        SOURCES test_krylov.cpp)
  ENDIF()
endif()

target_link_libraries(krylov_example.exe PUBLIC
                      Alien::alien_core
                      Alien::alien_semantic_ref
                      arcconpkg_MPI
                      ${Boost_LIBRARIES})


#-----------------------------------------------------------
#
#  Test solver on Block Matrices
add_executable(block_solver_example.exe test_block_solver.cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID STREQUAL Clang))
  if(ALIEN_WANT_AVX)
    target_compile_options(block_solver_example.exe INTERFACE -mavx)
  endif()
  if(ALIEN_WANT_AVX2)
    target_compile_options(block_solver_example.exe INTERFACE -mavx -mfma)
  endif()
  if(ALIEN_WANT_AVX512)
    target_compile_options(block_solver_example.exe INTERFACE -mavx512f -mavx512cd)
  endif()
endif()

if(ALIEN_USE_SYCL)
  target_compile_definitions(block_solver_example.exe PUBLIC ALIEN_USE_SYCL USE_SYCL2020)
  IF(ALIEN_USE_HIPSYCL)
    add_sycl_to_target(TARGET  block_solver_example.exe
                        SOURCES test_block_solver.cpp)
  ENDIF()
  IF(ALIEN_USE_INTELSYCL)
    #set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
    IF(ALIEN_USE_CUDA)
      set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
      target_compile_options(block_solver_example.exe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
    ENDIF()
    add_sycl_to_target(TARGET  block_solver_example.exe
                        SOURCES test_block_solver.cpp)
  ENDIF()
  IF(ALIEN_USE_ACPPSYCL)
    IF(CMAKE_COMPILER_IS_GNUCXX)
      target_compile_options(block_solver_example.exe PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
    ENDIF()
    target_compile_definitions(block_solver_example.exe PRIVATE USE_ACPPSYCL)
    add_sycl_to_target(TARGET  block_solver_example.exe
                        SOURCES test_block_solver.cpp)
  ENDIF()
endif()

target_link_libraries(block_solver_example.exe PUBLIC
                      Alien::alien_core
                      Alien::alien_semantic_ref
                      arcconpkg_MPI
                      ${Boost_LIBRARIES})

include(LoadAlienTest)

#-----------------------------------------------------------
# Tests
#-----------------------------------------------------------

alien_test( BENCH krylov-simplecsr
            NAME bicgs-diag
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond diag --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-diag-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond diag --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-cheb
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond cheb --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-cheb-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond cheb --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-poly
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond poly --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-poly-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond poly --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-ilu0
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond ilu0 --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-ilu0-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond ilu0 --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-filu0-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond filu0 --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME bicgs-filu0
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver bicgs --precond filu0 --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME cg-diag
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond diag --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME cg-cheb
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond cheb --kernel simplcsr)

alien_test( BENCH krylov-simplecsr
            NAME cg-poly
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond poly --kernel simplcsr)

if(ALIEN_USE_SYCL)

alien_test( BENCH sycl
            NAME sycl
            COMMAND test_sycl.exe)

alien_test( BENCH sycl
            NAME sycl-mpi
            PROCS 4
            COMMAND test_sycl.exe)

alien_test( BENCH sycl-mvhandler
            NAME sycl-mvhandler
            COMMAND test_mvhandler_sycl.exe)

alien_test( BENCH sycl-combine-mvhandler
            NAME sycl-combine-mvhandler
            COMMAND test_combine_mvhandler_sycl.exe)

alien_test( BENCH krylov-sycl
            NAME diag-bicgs
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond diag --kernel sycl --asynch 0)

alien_test( BENCH krylov-sycl
            NAME diag-bicgs-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond diag --kernel sycl --asynch 0)

alien_test( BENCH krylov-sycl
            NAME cheb-bicgs
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond cheb --kernel sycl --asynch 0)

alien_test( BENCH krylov-sycl
            NAME cheb-bicgs-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond cheb --kernel sycl --asynch 0)

alien_test( BENCH krylov-sycl
            NAME poly-bicgs
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond poly --kernel sycl --asynch 0)

alien_test( BENCH krylov-sycl
            NAME filu0-bicgs
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond filu0 --kernel sycl --asynch 0)

alien_test( BENCH krylov-sycl
            NAME diag-bicgs-asynch
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond diag --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME diag-bicgs-asynch-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond diag --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME cheb-bicgs-asynch
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond cheb --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME cheb-bicgs-asynch-mpi
            PROCS 4
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond cheb --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME poly-bicgs-asynch
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond poly --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME filu0-bicgs-asynch
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --precond filu0 --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME cg-diag
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond diag --kernel sycl)

alien_test( BENCH krylov-sycl
            NAME cg-cheb
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond cheb --kernel sycl)

alien_test( BENCH krylov-sycl
            NAME cg-poly
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond poly --kernel sycl)

alien_test( BENCH krylov-sycl
            NAME cg-diag-asynch
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond diag --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME cg-cheb-asynch
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond cheb --kernel sycl --asynch 1)

alien_test( BENCH krylov-sycl
            NAME cg-poly-asynch
            COMMAND krylov_example.exe
            OPTIONS --nx 10 --ny 10 --solver cg --precond poly --kernel sycl --asynch 1)
endif()
