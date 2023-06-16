cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
set(BUILD_SHARED_LIBS TRUE)
if (NOT Arcane_DIR)
  include(ArcaneCompilerConfig.cmake)
  message(STATUS "ArcaneRoot = ${Arcane_DIR}")
  message(STATUS "AxlstarRoot = ${Axlstar_DIR}")
  message(STATUS "ArcconRoot = ${Arccon_DIR}")
  message(STATUS "ArccoreRoot = ${Arccore_DIR}")
endif()
project(ArcaneSamples LANGUAGES C CXX)

# Si le répertoire d'installation est différent du répertoire
# des sources, il faut recopier les infos pour Nuget.
if(NOT CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  configure_file(nuget.config ${CMAKE_BINARY_DIR}/nuget.config)
  file(COPY _nuget_fallback DESTINATION ${CMAKE_BINARY_DIR})
endif()

enable_testing()

add_subdirectory(helloworld)
add_subdirectory(poisson)
add_subdirectory(microhydro)
add_subdirectory(eos)
add_subdirectory(geometry)
add_subdirectory(print_meshinfo)
add_subdirectory(direct_cartesian)
add_subdirectory(hydro_accelerator)
add_subdirectory(standalone_accelerator)
add_subdirectory(standalone_subdomain)
add_subdirectory(mini_nbody)
add_subdirectory(honeycomb_heat)
add_subdirectory(simple_csv_output)
add_subdirectory(multiple_mesh)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
