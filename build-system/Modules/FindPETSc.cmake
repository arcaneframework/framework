arccon_return_if_package_found(PETSc)

find_path(PETSC_INCLUDE_DIRS petsc.h)

find_library(PETSC_LIB petsc)
find_library(PETSC_KSP_LIB petscksp)
find_library(PETSC_MAT_LIB petscmat)
find_library(PETSC_VEC_LIB petscvec)
find_library(PETSC_DM_LIB petscdm)

if (PETSC_LIB)
  set(PETSC_LIBRARIES ${PETSC_LIB})
endif()

if (PETSC_KSP_LIB)
  list(APPEND PETSC_LIBRARIES ${PETSC_KSP_LIB})
endif()

if (PETSC_MAT_LIB)
  list(APPEND PETSC_LIBRARIES ${PETSC_MAT_LIB})
endif()

if (PETSC_VEC_LIB)
  list(APPEND PETSC_LIBRARIES ${PETSC_VEC_LIB})
endif()

if (PETSC_DM_LIB)
  list(APPEND PETSC_LIBRARIES ${PETSC_DM_LIB})
endif()

message(STATUS "PETSC_INCLUDE_DIRS=" ${PETSC_INCLUDE_DIRS})
message(STATUS "PETSC_LIBRARIES=" ${PETSC_LIBRARIES} )

unset(PETSc_FOUND)
if (PETSC_INCLUDE_DIRS AND PETSC_LIBRARIES)
  set(PETSc_FOUND TRUE)
  arccon_register_package_library(PETSc PETSC)
endif()
