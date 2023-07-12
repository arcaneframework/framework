#
# Find Fortran includes
#
# This module defines
# FORTRAN_LIBRARIES, the libraries to link against to use Fortran.
# FORTRAN_FOUND If false, do not try to use Fortran.
set(FORTRAN_LIBRARY_PATH $ENV{F90_ROOT})
IF(NOT WIN32)
  foreach(FORTRAN_LIB ifcore imf svml irc ifport)
    find_library(FORTRAN_SUB_${FORTRAN_LIB} ${FORTRAN_LIB}
      PATHS ${FORTRAN_LIBRARY_PATH}
      PATH_SUFFIXES lib lib/intel64
      NO_DEFAULT_PATH)
    if(FORTRAN_SUB_${FORTRAN_LIB})
      set(FORTRAN_LIBRARY ${FORTRAN_LIBRARY} ${FORTRAN_SUB_${FORTRAN_LIB}})
    else(FORTRAN_SUB_${FORTRAN_LIB})
      set(FORTRAN_LIBRARY_FAILED "YES")
    endif(FORTRAN_SUB_${FORTRAN_LIB})
  endforeach(FORTRAN_LIB)

else(NOT WIN32)
foreach(FORTRAN_LIB libifcoremt.lib libircmt.lib)
  find_library(FORTRAN_SUB_${FORTRAN_LIB} ${FORTRAN_LIB}
    PATHS ${FORTRAN_LIBRARY_PATH} NO_DEFAULT_PATH)
  if(FORTRAN_SUB_${FORTRAN_LIB})
    set(FORTRAN_LIBRARY ${FORTRAN_LIBRARY} ${FORTRAN_SUB_${FORTRAN_LIB}})
  else(FORTRAN_SUB_${FORTRAN_LIB})
    message(STATUS "Fortran library ${FORTRAN_SUB_${FORTRAN_LIB}} not found")
    set(FORTRAN_LIBRARY_FAILED "YES")
  endif(FORTRAN_SUB_${FORTRAN_LIB})
endforeach(FORTRAN_LIB)
endif(NOT WIN32)

set(FORTRAN_FOUND "NO")
if(NOT FORTRAN_LIBRARY_FAILED)
  set(FORTRAN_FOUND "YES")
  set(FORTRAN_LIBRARIES ${FORTRAN_LIBRARY})
  set(FORTRAN_FLAGS "_F2C" "USE_FORTRAN")
  message(status "FORTRAN LIBRARIES : ${FORTRAN_LIBRARIES}")

  add_library(fortran INTERFACE IMPORTED)
  
  foreach(FORTRAN_LIB ifcore imf svml irc ifport)
    add_library(${FORTRAN_LIB} UNKNOWN IMPORTED)

    set_target_properties(${FORTRAN_LIB} PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${FORTRAN_SUB_${FORTRAN_LIB}}")

    set_property(TARGET fortran APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "${FORTRAN_LIB}")

  endforeach(FORTRAN_LIB)

endif(NOT FORTRAN_LIBRARY_FAILED)
