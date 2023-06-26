option(ALIEN_PETSC_UNIT_TESTS "Enable Hypre wrapper tests" ${ALIEN_UNIT_TESTS})

find_package(MPI REQUIRED)

find_package(PETSc REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(petsc_wrapper
        include/alien/petsc/backend.h
        src/matrix.h
        src/matrix.cpp
        src/vector.h
        src/vector.cpp
        src/petsc_linear_algebra.cpp
        src/petsc_linear_solver.cpp
        src/converters/simplecsr_to_petsc_matrix.cpp
        src/converters/simplecsr_to_petsc_vector.cpp
        src/converters/petsc_to_simplecsr_vector.cpp
        src/petsc_instance.cpp src/petsc_instance.h)

target_link_libraries(petsc_wrapper PUBLIC
        Alien::alien_core
        )

target_link_libraries(petsc_wrapper PRIVATE
        MPI::MPI_CXX
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi
        arccon::PETSc
        )

target_include_directories(petsc_wrapper PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

add_library(Alien::petsc_wrapper ALIAS petsc_wrapper)

include(GenerateExportHeader)

generate_export_header(petsc_wrapper
        BASE_NAME ALIEN_PETSC
        EXPORT_FILE_NAME include/alien/petsc/export.h
        )

# Tests
if (ALIEN_PETSC_UNIT_TESTS)
    enable_testing()
    add_subdirectory(test)
endif (ALIEN_PETSC_UNIT_TESTS)

# Install

include(GNUInstallDirs)

set(ALIENPETSC_EXPORT_TARGET ${PROJECT_NAME}Targets)

install(DIRECTORY include/alien/petsc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
        )

install(TARGETS petsc_wrapper EXPORT ${ALIENPETSC_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/include/alien/petsc/export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/petsc
)

install(EXPORT ${ALIENPETSC_EXPORT_TARGET}
        NAMESPACE ${namespace}
        DESTINATION ${config_install_dir})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
