arccon_return_if_package_found(OSMesa)

find_library(OSMesa_LIBRARY OSMesa)
message(STATUS "OSMesa_LIBRARY = ${OSMesa_LIBRARY}")

if (OSMesa_LIBRARY)
  get_filename_component(OSMesa_LIB_PATH ${OSMesa_LIBRARY} PATH)
  get_filename_component(OSMesa_ROOT_PATH ${OSMesa_LIB_PATH} PATH)
  message(STATUS "OSMesa ROOT PATH = ${OSMesa_ROOT_PATH}")
endif ()

find_path(OSMesa_INCLUDE_DIR GL/osmesa.h
  PATHS
  ${OSMesa_ROOT_PATH}/include
)

message(STATUS "OSMesa_INCLUDE_DIR = ${OSMesa_INCLUDE_DIR}")
 
set(OSMesa_FOUND FALSE)
if (OSMesa_INCLUDE_DIR AND OSMesa_LIBRARY)
  set(OSMesa_FOUND TRUE)
  set(OSMesa_LIBRARIES ${OSMesa_LIBRARY})
  set(OSMesa_INCLUDE_DIRS ${OSMesa_INCLUDE_DIR})
endif()
arccon_register_package_library(OSMesa OSMesa)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
