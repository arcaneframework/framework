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
find_package(Boost COMPONENTS program_options)

if(ALIEN_USE_SYCL)
add_executable(sycl_example.exe test_sycl.cpp)

target_link_libraries(sycl_example.exe PUBLIC
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
  target_compile_definitions(krylov_example.exe PUBLIC -DALIEN_USE_SYCL)
endif()
 
target_link_libraries(krylov_example.exe PUBLIC
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
            NAME default
            COMMAND sycl_example.exe)

alien_test( BENCH sycl
            NAME default-mpi
            PROCS 4
            COMMAND sycl_example.exe)
        
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
