# TODO: utiliser module cmake fourni par Mathematica pour trouver le package MathLink
include(srcs.cmake)

arcane_add_library(arcane_mathlink
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/mathlink
  FILES ${ARCANE_SOURCES}
)

target_link_libraries(arcane_mathlink PUBLIC arcane_core)
target_link_libraries(arcane_mathlink PRIVATE ${MATHLINK_LIBRARY})
target_include_directories(arcane_mathlink PRIVATE ${MATHLINK_INCLUDE_DIR})
target_include_directories(arcane_mathlink PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)

arcane_register_library(arcane_mathlink OPTIONAL)
