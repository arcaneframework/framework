cmake_minimum_required(VERSION 3.7.2)

# For support of VERSION
cmake_policy(SET CMP0048 NEW)

project(Arccon
  LANGUAGES NONE # No C or C++ !
  VERSION 1.0.0)

include(${CMAKE_CURRENT_LIST_DIR}/build-system/ArcconSetInstallDirs.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/build-system/commands/user/installDirectory.cmake)

set(config_install_dir "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}")
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(arccon_install_lib "${CMAKE_INSTALL_LIBDIR}/arccon")

arccon_install_directory(NAMES build-system packages DESTINATION ${arccon_install_lib})
install(FILES Arccon.cmake loadArccon.cmake Readme.md DESTINATION ${arccon_install_lib})

set(PACKAGES_DIR_SYSTEM ${arccon_install_lib}/build-system/packages)
set(PACKAGES_DIR_P ${arccon_install_lib}/packages)
set(MODULE_PATH_SYSTEM ${arccon_install_lib}/build-system)
set(MODULE_PATH_M ${arccon_install_lib}/build-system/Modules)

set(CMAKE_COMMANDS ${MODULE_PATH_SYSTEM}/commands/commands.cmake)

# Configuration
set(project_config_version "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")

include(CMakePackageConfigHelpers)
configure_package_config_file(ArcconConfig.cmake.in ${project_config}
  INSTALL_DESTINATION ${config_install_dir}
  PATH_VARS PACKAGES_DIR_SYSTEM PACKAGES_DIR_P
  MODULE_PATH_SYSTEM MODULE_PATH_M
  CMAKE_COMMANDS
  )
write_basic_package_version_file(
  ${project_config_version}
  VERSION ${Arccon_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES ${project_config_version} ${project_config}
        DESTINATION ${config_install_dir} )
