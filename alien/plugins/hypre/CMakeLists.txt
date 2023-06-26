option(ALIEN_HYPRE_UNIT_TESTS "Enable Hypre wrapper tests" ${ALIEN_UNIT_TESTS})
option(ALIEN_HYPRE_DEVICE "Enable hypre's GPU backend" OFF)

find_package(Hypre REQUIRED)

find_package(MPI REQUIRED)

include(CMakePushCheckState)
cmake_push_check_state()
get_target_property(HYPRE_TARGET_DEFINITIONS arccon::Hypre INTERFACE_COMPILE_DEFINITIONS)
get_target_property(HYPRE_TARGET_INCLUDES arccon::Hypre INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(HYPRE_TARGET_LINK_OPTIONS arccon::Hypre INTERFACE_LINK_OPTIONS)
get_target_property(HYPRE_TARGET_LIBRARIES arccon::Hypre INTERFACE_LINK_LIBRARIES)

macro(setCMakeRequired NAME)
    if (HYPRE_TARGET_${NAME})
        set(CMAKE_REQUIRED_${NAME} ${HYPRE_TARGET_${NAME}})
    endif ()
endmacro()

setCMakeRequired(DEFINITIONS)
setCMakeRequired(INCLUDES)
setCMakeRequired(LINK_OPTIONS)
setCMakeRequired(LIBRARIES)

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(HYPRE_Init "HYPRE_utilities.h" HAVE_HYPRE_INIT)

include(CheckTypeSize)
set(CMAKE_EXTRA_INCLUDE_FILES "HYPRE_utilities.h")
check_type_size(HYPRE_BigInt HYPRE_BIGINT)

check_type_size(HYPRE_MEMORY_DEVICE HYPRE_MEMORY_DEVICE)

if (ALIEN_HYPRE_DEVICE AND NOT HAVE_HYPRE_MEMORY_DEVICE)
    message(FATAL "Hypre device support enabled but not available.")
endif ()

cmake_pop_check_state()

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
        src/hypre_instance.h
        src/hypre_instance.cpp
        src/converters/hypre_to_simplecsr_vector.cpp
        src/converters/simplecsr_to_hypre_vector.cpp
        src/converters/simplecsr_to_hypre_matrix.cpp
        src/hypre_linear_solver.h)

if (ALIEN_HYPRE_DEVICE)
    target_compile_definitions(hypre_wrapper PRIVATE ALIEN_HYPRE_DEVICE)
endif ()

if (HAVE_HYPRE_INIT)
    target_compile_definitions(hypre_wrapper PRIVATE HAVE_HYPRE_INIT)
endif ()

if (HAVE_HYPRE_BIGINT)
    target_compile_definitions(hypre_wrapper PRIVATE HAVE_HYPRE_BIGINT)
endif ()


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

include(GNUInstallDirs)

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
