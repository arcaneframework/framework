# Find Zoltan includes and library
arccon_return_if_package_found(Zoltan)

# Ce package est fourni par trilinos. Il est aussi possible pour les anciennes
# versions (avant 2012) de le trouver directement.
set(_SAVED_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
unset(CMAKE_MODULE_PATH)
find_package(Zoltan2 QUIET)
set(CMAKE_MODULE_PATH ${_SAVED_CMAKE_MODULE_PATH})

message(STATUS "Zoltan2_FOUND? ${Zoltan2_FOUND}")
if (TARGET zoltan2)
  message(STATUS "HAS_TARGET_ZOLTAN")
  get_target_property(ZOLTAN_LIBRARIES zoltan2 IMPORTED_LOCATION)
  get_target_property(ZOLTAN_INCLUDE_DIRS zoltan2 INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "ZOLTAN2_LIBRARIES = ${Zoltan2_LIBRARIES}")
  message(STATUS "ZOLTAN2_INCLUDE_DIRS = ${Zoltan2_INCLUDE_DIRS}")
  set(Zoltan_LIBRARIES  ${Zoltan2_LIBRARIES})
  set(Zoltan_INCLUDE_DIRS ${Zoltan2_INCLUDE_DIRS})
  set(Zoltan_FOUND TRUE)
  arccon_register_package_library(Zoltan Zoltan)
  return()
endif()

# Si pas trouvé via trilinos, essaie de le trouver manuellement
arccon_find_legacy_package(NAME Zoltan LIBRARIES zoltan HEADERS zoltan.h)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
