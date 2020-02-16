if (NOT TARGET hypre AND NOT TARGET petsc AND NOT TARGET mtl)
    return()
endif ()

createLibrary(ALIEN-ExternalPackages
        EXPORT ALIEN/ALIEN-ExternalPackagesExport.h
        )

set_target_properties(ALIEN-ExternalPackages
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

if(TARGET hypre)
    addDirectory(ALIEN/Kernels/Hypre)
endif()

if(TARGET mtl)
    addDirectory(ALIEN/Kernels/MTL)
endif()

if(TARGET petsc)
    addDirectory(ALIEN/Kernels/PETSc)
    addSources(ALIEN-ExternalPackages
            ALIEN/Functional/Dump.cc
            )
endif()

if (TARGET arcane_core)
    linkLibraries(ALIEN-ExternalPackages arcane_core arcane_std)
else ()
    linkLibraries(ALIEN-ExternalPackages ALIEN-StrongOptions)
endif()

if(TARGET umfpack)
    linkLibraries(ALIEN-ExternalPackages umfpack)
endif()

linkLibraries(ALIEN-ExternalPackages ALIEN)

target_include_directories(ALIEN-ExternalPackages PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)

commit(ALIEN-ExternalPackages)

install(DIRECTORY ALIEN
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        )
