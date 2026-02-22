
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
  #find_package(IntelSYCL REQUIRED)
  message(status "IntelSYCL              : ${IntelSYCL_FOUND}")
  message(status "IntelSYCL FOUND        : ${IntelSYCL_FOUND}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
  message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
  message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
  message(status "      SYCL LINK FLAGS  : ${SYCL_LINK_FLAGS}")

ENDIF()

#createLibrary(alien_external_hypre_sycl
#        EXPORT alien/AlienExternalPackagesExport.h
#        )

add_library(alien_external_hypre_sycl SHARED OBJECT
        HCSR_to_Hypre_VectorConverter.cc
        HCSR_to_Hypre_MatrixConverter.cc
        SYCL_to_Hypre_MatrixConverter.cc
        )

set_target_properties(alien_external_hypre_sycl
        PROPERTIES LINKER_LANGUAGE "CXX")


#addSources(alien_external_hypre_sycl
#        HCSR_to_Hypre_VectorConverter.cc
#        HCSR_to_Hypre_MatrixConverter.cc
#        )

IF(ALIEN_USE_HIPSYCL)
  IF(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(alien_external_hypre_sycl PRIVATE "--gcc-toolchain=${GCCCORE_ROOT}")
  ENDIF()
  set(DPCPP_FLAGS --hipsycl-gpu-arch=gfx90a --hipsycl-platform=rocm)
  target_compile_options(alien_kernel_sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
  target_compile_definitions(alien_kernel_sycl PRIVATE USE_HIPSYCL)
  #target_include_directories(alien_kernel_sycl PRIVATE ${SYCL_INCLUDE_DIR_HINT})
  add_sycl_to_target(TARGET alien_external_hypre_sycl
                     SOURCES HCSR_to_Hypre_VectorConverter.cc
                             HCSR_to_Hypre_MatrixConverter.cc
                             SYCL_to_Hypre_MatrixConverter.cc)
ENDIF()

IF(ALIEN_USE_INTELSYCL)
   IF(ALIEN_USE_CUDA)
     set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -Wno-linker-warnings )
     target_compile_options(alien_external_hypre_sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
     target_link_options(alien_external_hypre_sycl PRIVATE ${DPCPP_FLAGS})
   ENDIF()
   IF(ALIEN_USE_HIP)
     set(DPCPP_FLAGS -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -Wno-linker-warnings)
     #set(DPCPP_FLAGS -fsycl --amdgpu-target=gfx90a --offload-arch=gfx90a -Wno-linker-warnings)
     target_compile_options(alien_external_hypre_sycl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>)
     target_link_options(alien_external_hypre_sycl PRIVATE ${DPCPP_FLAGS})
   ENDIF()
   add_sycl_to_target(TARGET  alien_external_hypre_sycl
                      SOURCES HCSR_to_Hypre_VectorConverter.cc
                              HCSR_to_Hypre_MatrixConverter.cc
                              SYCL_to_Hypre_MatrixConverter.cc)
ENDIF()


target_link_libraries(alien_external_hypre_sycl alien_core)

include_directories(${PROJECT_SOURCE_DIR}/modules/external_packages/src ${PROJECT_BINARY_DIR}/modules/external_packages/src)
message(status "INCLUDE PATH :${PROJECT_BINARY_DIR} ")
target_include_directories(alien_external_hypre_sycl PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)


#commit(alien_external_hypre_sycl)

install(DIRECTORY alien
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        )

# Explicitly force linking of dependent libraries.
# This is needed on some platforms (ubuntu) to prevent linker from removing
# library if its symbols are not explicitly used.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Adding 'no-as-needed' to link option for 'alien_external_packages'")
  target_link_options(alien_external_hypre_sycl INTERFACE "-Wl,--no-as-needed")
endif()
