if (NOT TARGET trilinos)
    return()
endif ()

createLibrary(ALIEN-Trilinos
        EXPORT ALIEN/ALIEN-TrilinosExport.h
        )

set_target_properties(ALIEN-Trilinos
        PROPERTIES LINKER_LANGUAGE "CXX")

if (TARGET arcane_core)
    set(AXL_OPTION_GENERATION_TYPE "ALL")
    set(NAMESPACE_TYPE "Arcane")
    set(IS_WITH_ARCANE "")
    set(IS_WITH_MESH "NO_MESH")
else ()
    set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "NO_ARCANE")
    set(IS_WITH_MESH "NO_MESH")
endif ()


if (TARGET trilinos)
    message(status "USE TRILINOS")
    addDirectory(ALIEN/Kernels/Trilinos)
endif ()

if (TARGET arcane_core)
    linkLibraries(ALIEN-Trilinos arcane_core arcane_std)
else ()
    linkLibraries(ALIEN-Trilinos ALIEN-StrongOptions)
endif ()

if (TARGET umfpack)
    linkLibraries(ALIEN-Trilinos umfpack)
endif ()

if(TARGET nvamg)
  linkLibraries(ALIEN-Trilinos nvamg)
endif()

linkLibraries(ALIEN-Trilinos ALIEN ALIEN-RefSemanticMVHandlers)

target_include_directories(ALIEN-Trilinos PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)

commit(ALIEN-Trilinos)

install(DIRECTORY ALIEN
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        )
