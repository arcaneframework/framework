option(ALIEN_HYPRE_UNIT_TESTS "Enable Hypre wrapper tests" ${ALIEN_UNIT_TESTS})

find_package(Hypre REQUIRED)

find_package(MPI REQUIRED)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(hypre_wrapper
        include/alien/hypre/backend.h
        include/alien/hypre/options.h
        src/hypre_vector.cpp
        src/hypre_vector.h
        src/hypre_matrix.cpp
        src/hypre_matrix.h
        src/hypre_linear_algebra.cpp
        src/hypre_linear_solver.cpp
        src/converters/hypre_to_simplecsr_vector.cpp
        src/converters/simplecsr_to_hypre_vector.cpp
        src/converters/simplecsr_to_hypre_matrix.cpp
        src/hypre_linear_solver.h)

target_link_libraries(hypre_wrapper PUBLIC
        Alien::alien_core
        )

target_link_libraries(hypre_wrapper PRIVATE
        MPI::MPI_CXX
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi
        arccon::Hypre
        )

target_include_directories(hypre_wrapper PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

add_library(Alien::hypre_wrapper ALIAS hypre_wrapper)

include(GenerateExportHeader)

generate_export_header(hypre_wrapper
        BASE_NAME ALIEN_HYPRE
        EXPORT_FILE_NAME include/alien/hypre/export.h
        )

# Examples
set(ALIEN_BENCHMARK ON)
add_subdirectory(examples)

# Tests
if (ALIEN_HYPRE_UNIT_TESTS)
    add_subdirectory(tests)
endif (ALIEN_HYPRE_UNIT_TESTS)

add_subdirectory(bench)

# Install

include(ArcconSetInstallDirs)

set(ALIENHYPRE_EXPORT_TARGET ${PROJECT_NAME}Targets)

install(DIRECTORY include/alien/hypre
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
        )

install(TARGETS hypre_wrapper EXPORT ${ALIENHYPRE_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/include/alien/hypre/export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien/hypre
)

install(EXPORT ${ALIENHYPRE_EXPORT_TARGET}
        NAMESPACE ${namespace}
        DESTINATION ${config_install_dir})

install(DIRECTORY examples/ DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples/hypre)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
