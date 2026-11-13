cmake_minimum_required(VERSION 3.21)
project(ArcCher LANGUAGES CXX C)

if (NOT Arccore_FOUND)
  find_package(Arccore REQUIRED)
endif ()

add_subdirectory(arccher/core)
add_subdirectory(arccher/utils)
add_subdirectory(arccher/run)

add_executable(arccher main.cc)

set_target_properties(arccher PROPERTIES
  OUTPUT_NAME ArcCher
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  INSTALL_RPATH "$ORIGIN/../lib"
)

target_include_directories(arccher PRIVATE
  .
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(arccher PRIVATE
  Arccore::arccore_trace
  Arccore::arccore_common
  Arccore::arccore_base
)


# ----------------------------------------------------------------------------
# Installation

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS arccher
  EXPORT ArcaNetTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
