cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

message(STATUS "CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR} CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}  CMAKE_VERSION=${CMAKE_VERSION}")

set(generated_config_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")



# Pour Arcane.
set(common_doc_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(axldoc_path ${CMAKE_BINARY_DIR}/bin/axldoc)
set(common_doc_install_dir ${CMAKE_BINARY_DIR}/share/common_doc)
configure_file(cmake/CommonDocConfig.cmake.in ${CMAKE_BINARY_DIR}/share/common_doc/cmake/CommonDocConfig.cmake @ONLY)
file(COPY
  cmake/CommonDocCommonVars.cmake
  cmake/CommonDocUserVars.cmake
  cmake/CommonDocDevVars.cmake
  DESTINATION ${CMAKE_BINARY_DIR}/share/common_doc/cmake
)

# Pour les projets basés sur Arcane.
set(common_doc_dir ${CMAKE_INSTALL_PREFIX}/share/common_doc)
set(axldoc_path ${CMAKE_INSTALL_PREFIX}/bin/axldoc)
set(common_doc_install_dir ${CMAKE_INSTALL_PREFIX}/share/common_doc)
configure_file(cmake/CommonDocConfig.cmake.in ${generated_config_dir}/CommonDocConfig.cmake @ONLY)

install(DIRECTORY theme/ DESTINATION share/common_doc/theme)
install(FILES generate_axl_infos.sh.in DESTINATION share/common_doc)

install(FILES
  ${generated_config_dir}/CommonDocConfig.cmake
  cmake/CommonDocCommonVars.cmake
  cmake/CommonDocUserVars.cmake
  cmake/CommonDocDevVars.cmake
  DESTINATION share/common_doc/cmake
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
