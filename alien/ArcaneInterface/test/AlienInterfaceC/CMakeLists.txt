
add_subdirectory(C)

if(ENABLE_FORTRAN)
  if(TARGET mpifort OR TARGET ifort)
    enable_language(Fortran)
    add_subdirectory(Fortran)
  endif()
endif()
