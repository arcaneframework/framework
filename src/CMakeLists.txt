add_library(alien_hypre
        hypre_vector.cpp
        hypre_matrix.cpp
        internal/hypre_internal.cpp
        internal/hypre_internal_linear_algebra.cpp
        internal/hypre_internal_linear_solver.cpp
        converters/hypre_to_simplecsr_vector.cpp
        converters/simplecsr_to_hypre_vector.cpp
        converters/simplecsr_to_hypre_matrix.cpp
        hypre_backend.h)

target_link_libraries(alien_hypre PUBLIC
        arccore_trace
        arccore_collections
        arccore_base
        arccore_message_passing_mpi
        Alien::ALIEN
        arcconpkg_Hypre)

target_include_directories(alien_hypre PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ALIEN/hypre>)

include(GenerateExportHeader)

generate_export_header(alien_hypre
        EXPORT_FILE_NAME ALIEN/hypre/export.h)
