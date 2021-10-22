# -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
if (NOT EXAMPLE_NAME)
  message(FATAL_ERROR "Variable EXAMPLE_NAME not defined")
endif()

find_package(Arcane)

include(${Arcane_DIR}/ArcaneDotNet.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/samples.utils.cmake)

add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}Module.cc main.cc ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_axl.h)

arcane_generate_axl(${EXAMPLE_NAME})
configure_file(${EXAMPLE_NAME}.config ${CMAKE_CURRENT_BINARY_DIR} @ONLY)
configure_file(${EXAMPLE_NAME}.arc ${CMAKE_CURRENT_BINARY_DIR} @ONLY)
arcane_add_arcane_libraries_to_target(${EXAMPLE_NAME})
target_include_directories(${EXAMPLE_NAME} PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
