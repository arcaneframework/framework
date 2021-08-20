#
# Find the libunwind includes and library
#
# This module defines
# LibUnwind_INCLUDE_DIR, where to find headers,
# LibUnwind_LIBRARIES, the libraries to link against to use libunwind.
# LibUnwind_FOUND, If false, do not try to use libunwind.

find_path(LibUnwind_INCLUDE_DIR libunwind.h)
 
find_library(LibUnwind_GENERIC_LIBRARY unwind-generic)
find_library(LibUnwind_LIBRARY unwind)
find_library(LIBLZMA_LIBRARY lzma)

# NOTE: la libunwind peut utiliser la lib lzma
# pour gérer les symboles de débug qui sont compressés.
# Comme on ne peut pas savoir facilement si la version installée
# de libunwind utilise lzma, on cherche cette bibliothèque et
# on la rajoute si on la trouve.

set(LibUnwind_FOUND)
if (LibUnwind_INCLUDE_DIR AND LibUnwind_LIBRARY AND LibUnwind_GENERIC_LIBRARY)
  set(LibUnwind_FOUND TRUE)
  set(LibUnwind_LIBRARIES ${LibUnwind_GENERIC_LIBRARY} ${LibUnwind_LIBRARY} )
  if (LIBLZMA_LIBRARY)
    list(APPEND LibUnwind_LIBRARIES ${LIBLZMA_LIBRARY} )
  endif (LIBLZMA_LIBRARY)
  set(LibUnwind_INCLUDE_DIRS ${LibUnwind_INCLUDE_DIR})
endif()

arccon_register_package_library(LibUnwind LibUnwind)

