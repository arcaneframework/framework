


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
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
    ENDIF()
    IF(ALIEN_USE_INTELSYCL)
      find_package(IntelSYCL REQUIRED)
      message(status "IntelSYCL : ${IntelDPCPP_FOUND}")
      message(status "IntelSYCL FOUND : ${IntelDPCPP_FOUND}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR_HINT}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
    ENDIF()
    IF(ALIEN_USE_ACPPSYCL)
      find_package(AdaptiveCpp REQUIRED)
      message(status "AdaptiveSYCL           : ${AdaptiveCpp_FOUND}")
      message(status "      SYCL INCLUDE DIR : ${SYCL_INCLUDE_DIR}")
      message(status "      SYCL FLAGS       : ${SYCL_FLAGS}")
      message(status "      SYCL LINK FLAGS  : ${SYCL_LINK_FLAGS}")
      find_package(Boost COMPONENTS fiber context REQUIRED)
    ENDIF()
ENDIF()


createLibrary(alien_core_solvers
  EXPORT alien/AlienCoreSolversExport.h
  )

set_target_properties(alien_core_solvers
		      PROPERTIES LINKER_LANGUAGE "CXX")

#Needed if no external packages are found !
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cc)
addSources(alien_core_solvers ${CMAKE_CURRENT_BINARY_DIR}/empty.cc)

if (TARGET arcane_core)
    set(AXL_OPTION_GENERATION_TYPE "ALL")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "")
    set(IS_WITH_MESH "NO_MESH")
else ()
    set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "NO_ARCANE")
    set(IS_WITH_MESH "NO_MESH")
endif ()


if(USE_AXLSTAR)
    include(${AXLSTAR_CMAKE_COMMANDS})
endif()

addDirectory(alien/kernels/common)

addDirectory(alien/kernels/simple_csr)

if(ALIEN_USE_SYCL)
  addDirectory(alien/kernels/sycl)
endif()


if (TARGET arcane_core)
    linkLibraries(alien_core_solvers arcane_core arcane_std)
else ()
    linkLibraries(alien_core_solvers alien_strong_options)
endif()

if (TARGET Alien::alien_core)
    linkLibraries(alien_core_solvers Alien::alien_core)
endif()

if (TARGET Alien::alien_sycl)
    linkLibraries(alien_core_solvers Alien::alien_sycl)
endif()

if (TARGET alien_kernel_sycl)
    linkLibraries(alien_core_solvers alien_kernel_sycl)
endif()

if (TARGET alien_handlers_sycl)
    linkLibraries(alien_core_solvers alien_handlers_sycl)
endif()


if (TARGET Alien::alien_semantic_ref)
    linkLibraries(alien_core_solvers Alien::alien_semantic_ref)
endif()

linkLibraries(alien_core_solvers 
              Boost::program_options
              BLAS::BLAS)

target_include_directories(alien_core_solvers PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

commit(alien_core_solvers)

add_library(Alien::core_solvers ALIAS alien_core_solvers)

install(DIRECTORY alien
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
  )
