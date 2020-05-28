#
# Find the ALIEN includes and library
#
# This module uses
# ALIEN_ROOT
#
# This module defines
# ALIEN_FOUND
# ALIEN_INCLUDE_DIRS
# ALIEN_LIBRARIES
#
# Target alien

if(USE_ALIEN_V0)

  message(STATUS "Found ALIEN V0 at common/ALIEN")

  add_definitions(-DUSE_ALIEN_V0)
  
  return()

endif()

if(NOT ALIEN_ROOT)
  set(ALIEN_ROOT $ENV{ALIEN_ROOT})
endif()

if(NOT ALIEN_FOUND)

  set(ALIEN_DIR ${ALIEN_ROOT}/lib/cmake)

  find_package(ALIEN)

endif()

if(ALIEN_FOUND)

  get_target_property(ALIEN_INCLUDE_DIRS ALIEN INTERFACE_INCLUDE_DIRECTORIES)

  string(TOUPPER ${CMAKE_BUILD_TYPE} type)

  get_target_property(ALIEN_LIBRARY ALIEN IMPORTED_LOCATION_${type})
  get_target_property(ALIEN_IFPEN_LIBRARY ALIEN-IFPEN IMPORTED_LOCATION_${type})
  get_target_property(ALIEN_EXTERNALS_LIBRARY ALIEN-Externals IMPORTED_LOCATION_${type})
  get_target_property(ALIEN_EXTERNALPACKAGES_LIBRARY ALIEN-ExternalPackages IMPORTED_LOCATION_${type})

  set(ALIEN_LIBRARIES ${ALIEN_LIBRARY}
                      ${ALIEN_IFPEN_LIBRARY}
                      ${ALIEN_EXTERNALS_LIBRARY}
                      ${ALIEN_EXTERNALPACKAGES_LIBRARY})

  add_library(alien INTERFACE IMPORTED)
  
  set_property(TARGET alien APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ALIEN")

  set_property(TARGET alien APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ALIEN-IFPEN")

  set_property(TARGET alien APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ALIEN-Externals")

  set_property(TARGET alien APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "ALIEN-ExternalPackages")

  import_targets(TARGET alien XML ${ALIEN_ROOT}/lib/pkglist.xml)
  
  add_definitions(-DUSE_ALIEN_V1)

else()
  
  add_definitions(-DUSE_ALIEN_ARCGEOSIM)

endif()
