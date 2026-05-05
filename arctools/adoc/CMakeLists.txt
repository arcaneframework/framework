cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(generated_config_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Pour Arcane.
set(adoc_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(axldoc_path ${CMAKE_BINARY_DIR}/bin/axldoc)
set(adoc_install_dir ${CMAKE_BINARY_DIR}/share/adoc)
configure_file(cmake/ADocConfig.cmake.in ${CMAKE_BINARY_DIR}/share/adoc/cmake/ADocConfig.cmake @ONLY)
file(COPY
  cmake/ADocCommonVars.cmake
  cmake/ADocUserVars.cmake
  cmake/ADocDevVars.cmake
  DESTINATION ${CMAKE_BINARY_DIR}/share/adoc/cmake
)

# Pour les projets basés sur Arcane.
set(adoc_dir ${CMAKE_INSTALL_PREFIX}/share/adoc)
set(axldoc_path ${CMAKE_INSTALL_PREFIX}/bin/axldoc)
set(adoc_install_dir ${CMAKE_INSTALL_PREFIX}/share/adoc)
configure_file(cmake/ADocConfig.cmake.in ${generated_config_dir}/ADocConfig.cmake @ONLY)

install(DIRECTORY theme/ DESTINATION share/adoc/theme)
install(DIRECTORY doc/ DESTINATION share/adoc/doc)
install(FILES generate_axl_infos.sh.in DESTINATION share/adoc)

install(FILES
  ${generated_config_dir}/ADocConfig.cmake
  cmake/ADocCommonVars.cmake
  cmake/ADocUserVars.cmake
  cmake/ADocDevVars.cmake
  DESTINATION share/adoc/cmake
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
