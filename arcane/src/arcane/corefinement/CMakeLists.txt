message(STATUS "Include 'corefinement'")
set(PKGS GeometryKernel)
arcane_find_package(GeometryKernel)
if (NOT GeometryKernel_FOUND)
  return()
endif()

message(STATUS "Adding component 'corefinement'")

include(srcs.cmake)

arcane_add_library(arcane_corefinement
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/corefinement
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

target_link_libraries(arcane_corefinement PUBLIC arcane_core ${GEOMETRYKERNEL_LIBRARIES})
target_include_directories(arcane_corefinement PUBLIC arcane_core ${GEOMETRYKERNEL_INCLUDE_DIRS})
