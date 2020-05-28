#
# Find Blas library
#

if(BLAS_LIBRARIES)
  set(BLAS_FOUND TRUE)
endif()

if(NOT BLAS_FOUND)
   # pour limiter le mode verbose
  set(BLAS_FIND_QUIETLY ON)
  find_package(BLAS)
endif()

if(NOT BLAS_FOUND)

  find_library(BLAS_LIBRARIES
    NAMES blas libblas.so.3
    )

  find_package_handle_standard_args(BLAS
  	DEFAULT_MSG
  	BLAS_LIBRARIES)

endif()

if(BLAS_FOUND)
  
  set(BLAS_CBLAS_HEADER)

  set(CMAKE_REQUIRED_LIBRARIES_BACK ${CMAKE_REQUIRED_LIBRARIES})
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  set(CMAKE_REQUIRED_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

  check_symbol_exists(cblas_saxpy cblas.h BLAS_HAVE_CBLAS_HEADER)
  if(BLAS_HAVE_CBLAS_HEADER)
    set(BLAS_CBLAS_HEADER cblas.h)
    message(STATUS "CBLAS is available, HEADER is ${BLAS_CBLAS_HEADER}")
  endif()

  if(BLAS_CBLAS_HEADER)
    set(BLAS_HAVE_CBLAS True)
  else()
    get_filename_component(BLAS_MODULE_DIR  ${CMAKE_CURRENT_LIST_FILE} PATH)
    set(CMAKE_REQUIRED_INCLUDES ${BLAS_MODULE_DIR})
    check_symbol_exists(cblas_saxpy _find_cblas.h BLAS_HAVE_CBLAS)
    if(BLAS_HAVE_CBLAS)
      message(STATUS "CBLAS is available, but without header.")
    endif()
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_BACK})
  endif()

  if(NOT TARGET blas)

    add_library(blas UNKNOWN IMPORTED)

    set_target_properties(blas PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${BLAS_LIBRARIES}")

  endif()

endif()
