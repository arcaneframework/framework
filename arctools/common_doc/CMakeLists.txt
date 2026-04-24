cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

message(STATUS "CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR} CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}  CMAKE_VERSION=${CMAKE_VERSION}")

set(generated_config_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")



# Pour Arcane.
set(common_doc_dir ${CMAKE_CURRENT_SOURCE_DIR})
configure_file(cmake/CommonDocConfig.cmake.in ${CMAKE_BINARY_DIR}/share/common_doc/CommonDocConfig.cmake @ONLY)

# Pour les projets basés sur Arcane.
set(common_doc_dir ${CMAKE_INSTALL_PREFIX}/share/common_doc)
configure_file(cmake/CommonDocConfig.cmake.in ${generated_config_dir}/CommonDocConfig.cmake @ONLY)

install(DIRECTORY theme/ DESTINATION share/common_doc/theme)
install(FILES ${generated_config_dir}/CommonDocConfig.cmake DESTINATION share/common_doc/cmake)
install(FILES generate_axl_infos.sh.in DESTINATION share/common_doc)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
