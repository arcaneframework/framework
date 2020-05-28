#
# Find the PETSC includes and library
#
# This module uses
# PETSC_ROOT
#
# This module defines
# PETSC_FOUND
# PETSC_INCLUDE_DIRS
# PETSC_LIBRARIES
#
# Target petsc

find_package(PETSc)
arccon_return_if_package_found(PETSc)

message(STATUS "Warning: using obsolete FindPETSc")

if(NOT PETSC_ROOT)
  set(PETSC_ROOT $ENV{PETSC_ROOT})
endif()

if(PETSC_ROOT)
  set(_PETSC_SEARCH_OPTS NO_DEFAULT_PATH)
else()
  set(_PETSC_SEARCH_OPTS)
endif()

if(NOT PETSC_FOUND)

  find_library(PETSC_LIBRARY
  NAMES petsc libpetsc
		HINTS ${PETSC_ROOT}
		PATH_SUFFIXES lib
		${_PETSC_SEARCH_OPTS}
  )
  mark_as_advanced(PETSC_LIBRARY)

  find_library(PETSC_KSP_LIBRARY
  NAMES petscksp libpetscksp
		HINTS ${PETSC_ROOT}
		PATH_SUFFIXES lib
		${_PETSC_SEARCH_OPTS}
  )
  mark_as_advanced(PETSC_KSP_LIBRARY)

  find_library(PETSC_VEC_LIBRARY
  NAMES petscvec libpetscvec
		HINTS ${PETSC_ROOT}
		PATH_SUFFIXES lib
		${_PETSC_SEARCH_OPTS}
  )
  mark_as_advanced(PETSC_VEC_LIBRARY)

  find_library(PETSC_MAT_LIBRARY
  NAMES petscmat libpetscmat
		HINTS ${PETSC_ROOT}
		PATH_SUFFIXES lib
		${_PETSC_SEARCH_OPTS}
  )
  mark_as_advanced(PETSC_MAT_LIBRARY)

  find_library(PETSC_DM_LIBRARY
  NAMES petscdm libpetscdm
		HINTS ${PETSC_ROOT}
		PATH_SUFFIXES lib
		${_PETSC_SEARCH_OPTS}
  )
  mark_as_advanced(PETSC_DM_LIBRARY)

  find_path(PETSC_INCLUDE_DIR petsc.h
  HINTS ${PETSC_ROOT}
		PATH_SUFFIXES include
  ${_PETSC_SEARCH_OPTS}
  )
  mark_as_advanced(PETSC_INCLUDE_DIR)

endif()

# pour limiter le mode verbose
set(PETSC_FIND_QUIETLY ON)
set(PETSCKSP_FIND_QUIETLY ON)
set(PETSCVEC_FIND_QUIETLY ON)
set(PETSCMAT_FIND_QUIETLY ON)
set(PETSCDM_FIND_QUIETLY ON)

find_package_handle_standard_args(PETSC
	DEFAULT_MSG
	PETSC_INCLUDE_DIR
	PETSC_LIBRARY)

find_package_handle_standard_args(PETSCKSP
	DEFAULT_MSG
	PETSC_KSP_LIBRARY)
find_package_handle_standard_args(PETSCVEC
	DEFAULT_MSG
	PETSC_VEC_LIBRARY)
find_package_handle_standard_args(PETSCMAT
	DEFAULT_MSG
	PETSC_MAT_LIBRARY)
find_package_handle_standard_args(PETSCDM
	DEFAULT_MSG
	PETSC_DM_LIBRARY)

if(PETSC_FOUND AND NOT TARGET petsc)

  set(PETSC_INCLUDE_DIRS ${PETSC_INCLUDE_DIR})

  set(PETSC_LIBRARIES ${PETSC_LIBRARY})

  # petsc main

  add_library(petsc_main UNKNOWN IMPORTED)

  set_target_properties(petsc_main PROPERTIES
	  INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")

	set_target_properties(petsc_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${PETSC_LIBRARY}")

   add_library(petsc INTERFACE IMPORTED)

   set_property(TARGET petsc APPEND PROPERTY
   INTERFACE_LINK_LIBRARIES "petsc_main")


  # petscksp

  if(PETSCKSP_FOUND)

    list(APPEND PETSC_LIBRARIES ${PETSC_KSP_LIBRARY})

    add_library(petscksp UNKNOWN IMPORTED)

    set_target_properties(petscksp PROPERTIES
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")

	  set_target_properties(petscksp PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${PETSC_KSP_LIBRARY}")

    set_property(TARGET petsc APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "petscksp")


  endif()

  # petscvec

  if(PETSCVEC_FOUND)

    list(APPEND PETSC_LIBRARIES ${PETSC_VEC_LIBRARY})

    add_library(petscvec UNKNOWN IMPORTED)

    set_target_properties(petscvec PROPERTIES
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")

	  set_target_properties(petscvec PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${PETSC_VEC_LIBRARY}")

      set_property(TARGET petsc APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "petscvec")
  endif()

  # petscmat

	  if(PETSCMAT_FOUND)

    list(APPEND PETSC_LIBRARIES ${PETSC_MAT_LIBRARY})

    add_library(petscmat UNKNOWN IMPORTED)

    set_target_properties(petscmat PROPERTIES
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")

	  set_target_properties(petscmat PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${PETSC_MAT_LIBRARY}")

      set_property(TARGET petsc APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "petscmat")

  endif()

  # petscdm

  if(PETSCDM_FOUND)

    list(APPEND PETSC_LIBRARIES ${PETSC_DM_LIBRARY})

    add_library(petscdm UNKNOWN IMPORTED)

    set_target_properties(petscdm PROPERTIES
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")

	  set_target_properties(petscdm PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${PETSC_DM_LIBRARY}")

      set_property(TARGET petsc APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "petscdm")


  endif()

endif()
