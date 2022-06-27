option(ALIEN_TRILINOS_UNIT_TESTS "Enable Trilinos wrapper tests" ${ALIEN_UNIT_TESTS})


find_package(MPI REQUIRED)


# ----------------------------------------------------------------------------
find_package(Trilinos 13 REQUIRED COMPONENTS Teuchos Kokkos KokkosKernels Belos Tpetra Ifpack2)

# Echo trilinos build info just for fun
MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("End of Trilinos details\n")

# Make sure to use same compilers and flags as Trilinos
#MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
string(REPLACE " " ";" Trilinos_CXX_COMPILER_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(trilinos_wrapper
        include/alien/trilinos/backend.h
        include/alien/trilinos/options.h
        include/alien/trilinos/trilinos_config.h
        src/trilinos_vector.cpp
        src/trilinos_vector.h
        src/trilinos_matrix.cpp
        src/trilinos_matrix.h
        src/trilinos_linear_algebra.cpp
        src/trilinos_linear_solver.cpp
        src/trilinos_linear_solver.h
        src/converters/trilinos_to_simplecsr_vector.cpp
        src/converters/simplecsr_to_trilinos_vector.cpp
        src/converters/simplecsr_to_trilinos_matrix.cpp
        src/trilinos_config.cpp
        )

set_property(TARGET trilinos_wrapper
        APPEND_STRING
        PROPERTY
        COMPILE_OPTIONS ${Trilinos_CXX_COMPILER_FLAGS}
        )

target_link_libraries(trilinos_wrapper PUBLIC
        Alien::alien_core
        )

target_link_libraries(trilinos_wrapper PRIVATE
        MPI::MPI_CXX
        Arccore::arccore_trace
        Arccore::arccore_collections
        Arccore::arccore_base
        Arccore::arccore_message_passing_mpi
        ${Trilinos_LIBRARIES}
        ${Trilinos_TPL_LIBRARIES}
        )

if (ALIEN_PLUGIN_TRILINOS_OMP)
    message("------ COMPILING TRILINOS WRAPPER WITH *OPENMP* BACKEND  ------- ")
elseif (ALIEN_PLUGIN_TRILINOS_CUDA)
    message("------ COMPILING TRILINOS WRAPPER WITH *CUDA* BACKEND  ------- ")
elseif (ALIEN_PLUGIN_TRILINOS_HIP)
    message("------ COMPILING TRILINOS WRAPPER WITH *HIP* BACKEND  ------- ")
elseif (ALIEN_PLUGIN_TRILINOS_SERIAL)
    message("------ COMPILING TRILINOS WRAPPER WITH *SERIAL* BACKEND  ------- ")
elseif (ALIEN_PLUGIN_TRILINOS_THREADS)
    message("------ COMPILING TRILINOS WRAPPER WITH *PTHREAD* BACKEND  ------- ")
elseif (ALIEN_PLUGIN_TRILINOS_SYCL)
    message("------ COMPILING TRILINOS WRAPPER WITH *SYCL* BACKEND  ------- ")
else ()
    message(FATAL_ERROR "------ NO HARDWARE EXECUTOR SPECIFIED FOR TRILINOS WRAPPER  ------- ")
endif ()

configure_file(
        Trilinos.backend.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/alien/trilinos/TrilinosBackend.h
)

target_include_directories(trilinos_wrapper PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${Trilinos_INCLUDE_DIRS}
        ${Trilinos_TPL_INCLUDE_DIRS}
        )

add_library(Alien::trilinos_wrapper ALIAS trilinos_wrapper)

include(GenerateExportHeader)

generate_export_header(trilinos_wrapper
        BASE_NAME ALIEN_TRILINOS
        EXPORT_FILE_NAME include/alien/trilinos/export.h
        )

# Examples
set(ALIEN_BENCHMARK ON)
add_subdirectory(examples)

# Tests
if (ALIEN_TRILINOS_UNIT_TESTS)
    add_subdirectory(tests)
endif (ALIEN_TRILINOS_UNIT_TESTS)

# Install

include(GNUInstallDirs)

set(ALIENTRILINOS_EXPORT_TARGET ${PROJECT_NAME}Targets)

install(DIRECTORY include/alien/trilinos
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alien
        )

install(TARGETS trilinos_wrapper EXPORT ${ALIENTRILINOS_EXPORT_TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(
        FILES
        ${CMAKE_CURRENT_BINARY_DIR}/include/alien/trilinos/export.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/alien/trilinos/TrilinosBackend.h
        DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/alien/trilinos
)

install(EXPORT ${ALIENTRILINOS_EXPORT_TARGET}
        NAMESPACE ${namespace}
        DESTINATION ${config_install_dir})

install(DIRECTORY examples/ DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples/trilinos)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
