#
# Find the MKL includes and library
#
# This module uses
# MKL_ROOT
#
# This module defines
# MKL_FOUND
# MKL_INCLUDE_DIRS
# MKL_LIBRARIES
#
# Target mkl lapack cblas

if(NOT MKL_ROOT)
  set(MKL_ROOT $ENV{MKL_ROOT})
endif()

if(MKL_ROOT)
  set(_MKL_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_MKL_SEARCH_OPTS)
endif()

if(NOT MKL_FOUND)

  find_library(MKL_CORE_LIBRARY
    NAMES mkl_core
    HINTS ${MKL_ROOT}
		PATH_SUFFIXES lib/intel64 lib/em64t
    ${_MKL_SEARCH_OPTS}
    )
  mark_as_advanced(MKL_CORE_LIBRARY)

  find_library(MKL_LP64_LIBRARY
    NAMES mkl_intel_lp64
    HINTS ${MKL_ROOT}
		PATH_SUFFIXES lib/intel64 lib/em64t
    ${_MKL_SEARCH_OPTS}
    )
  mark_as_advanced(MKL_LP64_LIBRARY)

  find_library(MKL_SEQ_LIBRARY
    NAMES mkl_sequential
    HINTS ${MKL_ROOT}
		PATH_SUFFIXES lib/intel64 lib/em64t
    ${_MKL_SEARCH_OPTS}
    )
  mark_as_advanced(MKL_SEQ_LIBRARY)

  find_library(MKL_SCALP_LIBRARY
    NAMES mkl_scalapack_lp64
    HINTS ${MKL_ROOT}
		PATH_SUFFIXES lib/intel64 lib/em64t
    ${_MKL_SEARCH_OPTS}
    )
  mark_as_advanced(MKL_SCALP_LIBRARY)

  find_library(MKL_BLACS_LIBRARY
    NAMES mkl_blacs_intelmpi_lp64
    HINTS ${MKL_ROOT}
		PATH_SUFFIXES lib/intel64 lib/em64t
    ${_MKL_SEARCH_OPTS}
    )
  mark_as_advanced(MKL_BLACS_LIBRARY)

  find_path(MKL_INCLUDE_DIR mkl.h
    HINTS ${MKL_ROOT}
		PATH_SUFFIXES include
    ${_MKL_SEARCH_OPTS}
    )
  mark_as_advanced(MKL_INCLUDE_DIR)

endif()

# pour limiter le mode verbose
set(MKL_FIND_QUIETLY ON)

find_package_handle_standard_args(MKL
	DEFAULT_MSG
	MKL_INCLUDE_DIR
	MKL_CORE_LIBRARY
	MKL_LP64_LIBRARY
	MKL_SEQ_LIBRARY
	MKL_SCALP_LIBRARY
	MKL_BLACS_LIBRARY)

if(MKL_FOUND AND NOT TARGET mkl)

  set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
  
  set(MKL_LIBRARIES ${MKL_CORE_LIBRARY}
                    ${MKL_LP64_LIBRARY}
                    ${MKL_SEQ_LIBRARY}
                    ${MKL_SCALP_LIBRARY}
                    ${MKL_BLACS_LIBRARY})

  # mkl core
    
  add_library(mkl_core UNKNOWN IMPORTED)
    
  set_target_properties(mkl_core PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_core PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_CORE_LIBRARY}")
  
  # mkl intel lp64
  
  add_library(mkl_intel_lp64 UNKNOWN IMPORTED)
  
  set_target_properties(mkl_intel_lp64 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_intel_lp64 PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_LP64_LIBRARY}")
  
  # mkl sequential
  
  add_library(mkl_sequential UNKNOWN IMPORTED)
  
  set_target_properties(mkl_sequential PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_sequential PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_SEQ_LIBRARY}")
  
  # mkl scalapack lp64
  
  add_library(mkl_scalapack_lp64 UNKNOWN IMPORTED)
  
  set_target_properties(mkl_scalapack_lp64 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_scalapack_lp64 PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_SCALP_LIBRARY}")
  
  # mkl blacs intelmpi lp64
  
  add_library(mkl_blacs_intelmpi_lp64 UNKNOWN IMPORTED)
  
  set_target_properties(mkl_blacs_intelmpi_lp64 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_blacs_intelmpi_lp64 PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_BLACS_LIBRARY}")
  
  # mkl 
  
  add_library(mkl INTERFACE IMPORTED)
  
  set_property(TARGET mkl APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "mkl_core")

  set_property(TARGET mkl APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "mkl_intel_lp64")

  set_property(TARGET mkl APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "mkl_sequential")

  set_property(TARGET mkl APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "mkl_scalapack_lp64")

  set_property(TARGET mkl APPEND PROPERTY 
    INTERFACE_LINK_LIBRARIES "mkl_blacs_intelmpi_lp64")
  
  if(NOT CBLAS_FOUND)
  
    find_path(CBLAS_INCLUDE_DIR mkl_cblas.h
      HINTS ${MKL_ROOT} 
      PATH_SUFFIXES include
      ${_MKL_SEARCH_OPTS}
      )
    mark_as_advanced(MKL_INCLUDE_DIR)
  
  endif()
 
  # pour limiter le mode verbose
  set(CBLAS_FIND_QUIETLY ON)

  find_package_handle_standard_args(CBLAS 
	  DEFAULT_MSG 
	  CBLAS_INCLUDE_DIR)

  if(CBLAS_FOUND AND NOT TARGET cblas)
	  
    set(CBLAS_INCLUDE_DIRS "${CBLAS_INCLUDE_DIR}")
  
    set(CBLAS_LIBRARIES "${MKL_LIBRARIES}")
  
    add_library(cblas INTERFACE IMPORTED)
    
    set_target_properties(cblas PROPERTIES 
      INTERFACE_INCLUDE_DIRECTORIES "${CBLAS_INCLUDE_DIRS}") 
	  
    set_property(TARGET cblas APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "mkl")

  endif()
  
  if(NOT LAPACK_FOUND)
  
    find_path(LAPACK_INCLUDE_DIR mkl_lapack.h
      HINTS ${MKL_ROOT} 
      PATH_SUFFIXES include
      ${_MKL_SEARCH_OPTS}
      )
    mark_as_advanced(LAPACK_INCLUDE_DIR)
  
  endif()
 
  # pour limiter le mode verbose
  set(LAPACK_FIND_QUIETLY ON)

  find_package_handle_standard_args(LAPACK
	  DEFAULT_MSG 
	  LAPACK_INCLUDE_DIR)

  if(LAPACK_FOUND AND NOT TARGET lapack)
	  
    set(LAPACK_INCLUDE_DIRS "${LAPACK_INCLUDE_DIR}")
  
    set(LAPACK_LIBRARIES "${MKL_LIBRARIES}")
  
    add_library(lapack INTERFACE IMPORTED)
    
    set_target_properties(lapack PROPERTIES 
      INTERFACE_INCLUDE_DIRECTORIES "${LAPACK_INCLUDE_DIRS}") 
	  
    set_property(TARGET lapack APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "mkl")

  endif()
  
  add_library(blas INTERFACE IMPORTED)
  set_property(TARGET blas APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "mkl")
  
endif()
