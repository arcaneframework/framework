#
# Find the LM include and libraries
#
# This module defines
# LM_INCLUDE_DIRS, where to find headers
# LM_LIBRARIES, the libraries to link against to use LM tools
# LM_FOUND, If false, do not try to use LM tools.
 
find_library(LM_LIBRARY
  NAMES
  LM_dyn-2.4.14.3-gcc.7.3.0-Dyn
  LM_dyn-2.4.14.2-gcc.6.2.1-Dyn
  LM-2.4.14.2-${ARCANE_NECDIST_GCC_VERSION}
  PATHS
  ${ARCANE_NECDIST_ROOT}/shlib
  ${ARCANE_NECDIST_ROOT}/lib
)
find_path(LM_INCLUDE_DIR machine_types.h
  PATHS
  ${ARCANE_NECDIST_ROOT}/share/include/machine_types-1.0.1
)

message(STATUS "LM_LIBRARY = ${LM_LIBRARY}")
message(STATUS "LM_INCLUDE_DIR = ${LM_INCLUDE_DIR}")

set(LM_FOUND "NO")
if (LM_LIBRARY AND LM_INCLUDE_DIR)
  set(LM_FOUND "YES")
  set(LM_LIBRARIES ${LM_LIBRARY})
  set(LM_INCLUDE_DIRS ${LM_INCLUDE_DIR})
  arcane_add_package_library(lm LM)
endif()
