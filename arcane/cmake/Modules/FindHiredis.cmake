#
# Find the 'hiredis' library
#
# 'hiredis' use a cmake config file
#
arccon_return_if_package_found(hiredis)

# Supprime temporairement CMAKE_MODULE_PATH pour éviter une récursion
# infinie.

set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(hiredis)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

if (TARGET hiredis::hiredis)
  arccon_register_cmake_config_target(Hiredis CONFIG_TARGET_NAME hiredis::hiredis)
  return()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
