#
# Find the Intel MKL includes and library
#
# This module defines
# MKL_INCLUDE_DIRS, where to find headers,
# MKL_LIBRARIES, the libraries to link against to use MKL.
# MKL_FOUND, If false, do not try to use MKL.

arccon_return_if_package_found(MKL)

find_path(MKL_INCLUDE_DIRS mkl.h)
if (MKL_INCLUDE_DIRS)
  set( MKL_FOUND YES )
endif(MKL_INCLUDE_DIRS) 
foreach(MKL_LIBNAME mkl_rt mkl_intel_lp64 mkl_core mkl_sequential svml intlc)
  find_library(MKL_LIB_${MKL_LIBNAME} ${MKL_LIBNAME})
  if (MKL_LIB_${MKL_LIBNAME})
    message(STATUS "Found MKL library '${MKL_LIB_${MKL_LIBNAME}}'")
    list(APPEND MKL_LIBRARIES ${MKL_LIB_${MKL_LIBNAME}})
  else()
    message(STATUS "Missing MKL library '${MKL_LIBNAME}'")
    set(MKL_FOUND NO)
  endif()
endforeach()

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
message(STATUS "MKL_INCLUDE_DIRS = ${MKL_INCLUDE_DIRS}")
message(STATUS "MKL_LIBRARIES = ${MKL_LIBRARIES}")

arccon_register_package_library(MKL MKL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
