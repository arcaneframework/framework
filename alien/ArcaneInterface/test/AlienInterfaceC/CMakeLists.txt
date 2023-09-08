
add_subdirectory(C)

if (TARGET mpifort)
    add_subdirectory(Fortran)
endif()
