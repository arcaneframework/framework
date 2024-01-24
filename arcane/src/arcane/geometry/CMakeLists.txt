file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/arcane/geometry)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/arcane/geometry/euclidian)

include(srcs.cmake)

arcane_add_library(arcane_geometry
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/geometry
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

target_link_libraries(arcane_geometry PUBLIC arcane_core)

arcane_register_library(arcane_geometry)
