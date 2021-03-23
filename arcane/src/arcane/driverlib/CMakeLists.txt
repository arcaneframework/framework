include(srcs.cmake)

arcane_add_library(arcane_driverlib
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/driverlib
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_driverlib PRIVATE ARCANE_COMPONENT_arcane_driverlib)

target_link_libraries(arcane_driverlib ${ARCANE_DRIVERLIB_COMPONENT_DEPENDS})
target_link_libraries(arcane_driverlib arcane_std arcane_core)
arcane_register_library(arcane_driverlib OPTIONAL)
