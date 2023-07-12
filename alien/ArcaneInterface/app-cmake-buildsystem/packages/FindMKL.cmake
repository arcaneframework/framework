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
  
  if(NOT WIN32)
    add_library(mkl_core SHARED IMPORTED)
  else()
    add_library(mkl_core UNKNOWN IMPORTED)
  endif()

  set_target_properties(mkl_core PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  if(NOT WIN32)
    set_target_properties(mkl_core PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${MKL_CORE_LIBRARY}"
      IMPORTED_NO_SONAME ON)
  else()
    set_target_properties(mkl_core PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${MKL_CORE_LIBRARY}")
  endif()

  # mkl intel lp64
  
  if(NOT WIN32)
    add_library(mkl_intel_lp64 SHARED IMPORTED)
  else()
    add_library(mkl_intel_lp64 UNKNOWN IMPORTED)
  endif()
  
  set_target_properties(mkl_intel_lp64 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_intel_lp64 PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_LP64_LIBRARY}"
    IMPORTED_NO_SONAME ON)
  
  # mkl sequential

  if(NOT WIN32)
    add_library(mkl_sequential SHARED IMPORTED)
  else()
    add_library(mkl_sequential UNKNOWN IMPORTED)
  endif()

  set_target_properties(mkl_sequential PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_sequential PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_SEQ_LIBRARY}"
    IMPORTED_NO_SONAME ON)
  
  # mkl scalapack lp64

  if(NOT WIN32)
    add_library(mkl_scalapack_lp64 SHARED IMPORTED)
  else()
    add_library(mkl_scalapack_lp64 UNKNOWN IMPORTED)
  endif()
  
  set_target_properties(mkl_scalapack_lp64 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_scalapack_lp64 PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_SCALP_LIBRARY}"
    IMPORTED_NO_SONAME ON)
  
  # mkl blacs intelmpi lp64
  
  if(NOT WIN32)
    add_library(mkl_blacs_intelmpi_lp64 SHARED IMPORTED)
  else()
    add_library(mkl_blacs_intelmpi_lp64 UNKNOWN IMPORTED)
  endif() 

  set_target_properties(mkl_blacs_intelmpi_lp64 PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}") 
  
  set_target_properties(mkl_blacs_intelmpi_lp64 PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${MKL_BLACS_LIBRARY}"
    IMPORTED_NO_SONAME ON)
  
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
  
  # CBlas

  if(NOT CBLAS_FOUND)
    
    find_path(CBLAS_INCLUDE_DIR mkl_cblas.h
      HINTS ${MKL_ROOT} 
      PATH_SUFFIXES include
      ${_MKL_SEARCH_OPTS}
      )
    mark_as_advanced(CBLAS_INCLUDE_DIR)
    
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
  
  # Lapack
 
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
  
  # Blas

  add_library(blas INTERFACE IMPORTED)
  
  set_property(TARGET blas APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "mkl")
  
  # FFTW3

  if(NOT FFTW3_FOUND)
    
    find_library(MKL_FFTW3_LIBRARY
      NAMES fftw3x_cdft_lp64_pic
      HINTS ${MKL_ROOT} 
		  PATH_SUFFIXES lib/intel64 lib/em64t
      ${_MKL_SEARCH_OPTS}
      )
    mark_as_advanced(MKL_FFTW3_LIBRARY)

    find_path(FFTW3_INCLUDE_DIR fftw3.h
      HINTS ${MKL_ROOT} 
      PATH_SUFFIXES include include/fftw
      ${_MKL_SEARCH_OPTS}
      )
    mark_as_advanced(FFTW3_INCLUDE_DIR)
    
  endif()
  
  # pour limiter le mode verbose
  set(FFTW3_FIND_QUIETLY ON)

  find_package_handle_standard_args(FFTW3 
	  DEFAULT_MSG 
	  FFTW3_INCLUDE_DIR
    MKL_FFTW3_LIBRARY)

  if(FFTW3_FOUND AND NOT TARGET fftw3)
	  
    set(FFTW3_INCLUDE_DIRS "${FFTW3_INCLUDE_DIR}")
    
    set(FFTW3_LIBRARIES "${MKL_FFTW3_LIBRARY};${MKL_LIBRARIES}")
    
    if(NOT WIN32)
      add_library(mkl_fftw3_lp64 SHARED IMPORTED)
    else()
      add_library(mkl_fftw3_lp64 UNKNOWN IMPORTED)
    endif()
    
    set_target_properties(mkl_fftw3_lp64 PROPERTIES 
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}") 
  
    set_target_properties(mkl_fftw3_lp64 PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${MKL_FFTW3_LIBRARY}"
      IMPORTED_NO_SONAME ON)
    
    add_library(fftw3 INTERFACE IMPORTED)
    
    set_target_properties(fftw3 PROPERTIES 
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}") 

	  set_property(TARGET fftw3 APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "mkl")

    set_property(TARGET fftw3 APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "mkl_fftw3_lp64")

  endif()
  
endif()
