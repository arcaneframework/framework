# Set the minimum CMake required version
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# Add packages and modules provided by Arccon to the CMAKE_MODULE_PATH

get_filename_component(ARCCON_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(ARCCON_PACKAGE_DIRS ${ARCCON_PREFIX_DIR}/build-system/packages ${ARCCON_PREFIX_DIR}/packages)
list(APPEND CMAKE_MODULE_PATH ${ARCCON_PREFIX_DIR}/build-system ${ARCCON_PREFIX_DIR}/build-system/Modules)

include(${ARCCON_PREFIX_DIR}/build-system/commands/commands.cmake)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
