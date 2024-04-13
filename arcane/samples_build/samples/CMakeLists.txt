cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
set(BUILD_SHARED_LIBS TRUE)
if (NOT Arcane_DIR)
  if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/ArcaneCompilerConfig.cmake")
    include(ArcaneCompilerConfig.cmake)
  endif()
  message(STATUS "ArcaneRoot = ${Arcane_DIR}")
  message(STATUS "AxlstarRoot = ${Axlstar_DIR}")
  message(STATUS "ArcconRoot = ${Arccon_DIR}")
  message(STATUS "ArccoreRoot = ${Arccore_DIR}")
endif()
project(ArcaneSamples LANGUAGES C CXX)

find_package(Arcane REQUIRED)
arcane_accelerator_enable()

message(STATUS "Arcane installation prefix is '${ARCANE_PREFIX_DIR}'")

# Recopie les fichiers nécessaires pour nuget
if(NOT CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  file(COPY ${ARCANE_PREFIX_DIR}/samples/nuget.config DESTINATION ${CMAKE_BINARY_DIR})
  file(COPY ${ARCANE_PREFIX_DIR}/samples/_nuget_fallback DESTINATION ${CMAKE_BINARY_DIR})
endif()

enable_testing()

add_subdirectory(helloworld)
add_subdirectory(poisson)
add_subdirectory(microhydro)
add_subdirectory(eos)
add_subdirectory(geometry)
add_subdirectory(print_meshinfo)
add_subdirectory(direct_cartesian)
if(ARCANE_HAS_ACCELERATOR_API)
  add_subdirectory(hydro_accelerator)
  add_subdirectory(standalone_accelerator)
  add_subdirectory(mini_nbody)
endif()
add_subdirectory(standalone_subdomain)
add_subdirectory(honeycomb_heat)
add_subdirectory(simple_csv_output)
add_subdirectory(multiple_mesh)
if (ARCANE_HAS_DOTNET_WRAPPER)
  add_subdirectory(user_function)
endif()
add_subdirectory(simple_postprocessor)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
