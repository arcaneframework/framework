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
  

  if(PETSc_USE_PKGCONFIG)
     find_package(PkgConfig)
     set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
     list(APPEND CMAKE_PREFIX_PATH ${PETSC_ROOT}/lib/pkgconfig)
     set(PKG_CONFIG_PATH ${PETSC_ROOT}/lib/pkgconfig)
     pkg_check_modules(PKG_PETSC PETSc)
     message(STATUS "Infos from pkg_check_modules")
	 message(STATUS "PKGCONFIG PATH               = ${CMAKE_PREFIX_PATH}")
     message(STATUS "PKG_PETSC_INCLUDE_DIRS       = ${PKG_PETSC_INCLUDE_DIRS}")
     message(STATUS "PKG_PETSC_LIBRARIES          = ${PKG_PETSC_LIBRARIES}")
     message(STATUS "PKG_PETSC_LIBRARIES          = ${PKG_PETSC_STATIC_LIBRARIES}")
     message(STATUS "PKG_PETSC_LINK_LIBRARIES     = ${PKG_PETSC_LINK_LIBRARIES}")
     message(STATUS "PKG_PETSC_LDFLAG             = ${PKG_PETSC_LDFLAGS}")
     message(STATUS "PKG_PETSC_STATIC_LDFLAG      = ${PKG_PETSC_STATIC_LDFLAGS}")
     message(STATUS "PKG_PETSC_LIBRARY_DIRS       = ${PKG_PETSC_LIBRARY_DIRS}")
  endif()
  
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

  include(CheckPrototypeDefinition)
  
  check_prototype_definition(
    VecDestroy 
    "PetscErrorCode VecDestroy(Vec* src);" 0 "petsc.h;petscvec.h"
    PETSC_DESTROY_NEW
  )
  
  check_prototype_definition(
    KSPSetPCSide
    "PetscErrorCode KSPSetPCSide(KSP p1,PCSide p2);" 0 "petsc.h;petscksp.h"
    PETSC_KSPSETPCSIDE_NEW
  )
  
  check_prototype_definition(
    KSPSetOperators
    "PetscErrorCode KSPSetOperators(KSP p1,Mat m1,Mat m2);" 0 "petsc.h;petscksp.h"
    PETSC_KSPSETOPERATORS_NEW
  )
  
  check_prototype_definition(
    KSPDestroy 
    "PetscErrorCode KSPDestroy(KSP* src);" 0 "petsc.h;petscksp.h"
    PETSC_KSPDESTROY_NEW
    )
  
  check_prototype_definition(
    MatDestroy 
    "PetscErrorCode MatDestroy(Mat* src);" 0 "petsc.h;petscmat.h" 
    PETSC_MATDESTROY_NEW
    )
  
  check_prototype_definition(
    MatValid 
    "PetscErrorCode MatValid(Mat src, PetscTruth *flag);" 0 "petsc.h;petscmat.h" 
    PETSC_HAVE_MATVALID
    )
  
  check_prototype_definition(
    PetscViewerDestroy 
    "PetscErrorCode PetscViewerDestroy(PetscViewer* src);" 0 "petsc.h;petscviewer.h" 
    PETSC_VIEWERDESTROY_NEW
    )
  
  check_prototype_definition(
    KSPRichardsonSetSelfScale 
    "PetscErrorCode KSPRichardsonSetSelfScale(KSP src,PetscBool flag);" 0 "petsc.h;petscksp.h"
    PETSC_HAVE_KSPRICHARDSONSETSELFSCALE
    )
  
  check_prototype_definition(
    PCGetType
    "PetscErrorCode PCGetType(PC pc, PCType* pctype);" 0 "petsc.h;petscksp.h;petscpc.h"
    PETSC_GETPCTYPE_NEW
    )

  check_prototype_definition(
    PetscOptionsSetValue
    "PetscErrorCode PetscOptionsSetValue(PetscOptions options,const char name[],const char value[]);" 0 "petsc.h;petscsys.h"
    PETSC_OPTIONSSETVALUE_NEW
    )
        
  check_prototype_definition(
     PetscViewerAndFormatCreate
     "PetscErrorCode PetscViewerAndFormatCreate(PetscViewer viewer, PetscViewerFormat format, PetscViewerAndFormat **vf);" 0 "petsc.h;petscksp.h;petscviewer.h"
     PETSC_HAVE_VIEWERANDFORMAT
     )

  check_prototype_definition(
     KSPSetNullSpace
     "PetscErrorCode KSPSetNullSpace(KSP ksp,MatNullSpace nullsp);" 0 "petsc.h;petscksp.h"
     PETSC_HAVE_KSPSETNULLSPACE
     )

  message(status "CHECK PROTOTYPE PETSC_OPTIONSSETVALUE_NEW : ${PETSC_OPTIONSSETVALUE_NEW}")
  # petsc main
	  
  add_library(petsc_main UNKNOWN IMPORTED)
	  
  set_target_properties(petsc_main PROPERTIES 
	  INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")
  
	set_target_properties(petsc_main PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${PETSC_LIBRARY}")
  

  # petscksp
	  
  if(PETSCKSP_FOUND)
    
    list(APPEND PETSC_LIBRARIES ${PETSC_KSP_LIBRARY})

    add_library(petscksp UNKNOWN IMPORTED)
	  
    set_target_properties(petscksp PROPERTIES 
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")
    
	  set_target_properties(petscksp PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${PETSC_KSP_LIBRARY}")
    
  endif()
  
  # petscvec
	  
  if(PETSCVEC_FOUND)
    
    list(APPEND PETSC_LIBRARIES ${PETSC_VEC_LIBRARY})

    add_library(petscvec UNKNOWN IMPORTED)
	  
    set_target_properties(petscvec PROPERTIES 
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")
    
	  set_target_properties(petscvec PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${PETSC_VEC_LIBRARY}")
    
  endif()
  
  # petscmat
  
	  if(PETSCMAT_FOUND)
    
    list(APPEND PETSC_LIBRARIES ${PETSC_MAT_LIBRARY})

    add_library(petscmat UNKNOWN IMPORTED)
	  
    set_target_properties(petscmat PROPERTIES 
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")
    
	  set_target_properties(petscmat PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${PETSC_MAT_LIBRARY}")
    
  endif()
  
  # petscdm
	  
  if(PETSCDM_FOUND)
    
    list(APPEND PETSC_LIBRARIES ${PETSC_DM_LIBRARY})

    add_library(petscdm UNKNOWN IMPORTED)
	  
    set_target_properties(petscdm PROPERTIES 
	    INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIRS}")
    
	  set_target_properties(petscdm PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${PETSC_DM_LIBRARY}")
    
  endif()
  
  # petsc
  
  add_library(petsc INTERFACE IMPORTED)
  
  set_property(TARGET petsc APPEND PROPERTY 
               INTERFACE_LINK_LIBRARIES "petsc_main")
     

  if(PETSCKSP_FOUND)
    
    set_property(TARGET petsc APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "petscksp")
    
  endif()
  
  if(PETSCVEC_FOUND)
    
    set_property(TARGET petsc APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "petscvec")
    
  endif()
  
	  if(PETSCMAT_FOUND)
    
    set_property(TARGET petsc APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "petscmat")
    
  endif()
  
  if(PETSCDM_FOUND)
    
    set_property(TARGET petsc APPEND PROPERTY 
      INTERFACE_LINK_LIBRARIES "petscdm")
    
  endif()

  
endif()
