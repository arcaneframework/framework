createLibrary(alien_ifpen_solvers
  EXPORT alien/AlienIFPENSolversExport.h
  )

set_target_properties(alien_ifpen_solvers
		      PROPERTIES LINKER_LANGUAGE "CXX")

#Needed if no external packages are found !
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.cc)
addSources(alien_ifpen_solvers ${CMAKE_CURRENT_BINARY_DIR}/empty.cc)

if (TARGET arcane_core)
    set(AXL_OPTION_GENERATION_TYPE "ALL")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "")
    set(IS_WITH_MESH "NO_MESH")
else ()
    set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
    set(NAMESPACE_TYPE "Arccore")
    set(IS_WITH_ARCANE "NO_ARCANE")
    set(IS_WITH_MESH "NO_MESH")
endif ()

if(TARGET mcgsolver)
  addDirectory(ALIEN/Kernels/MCG)
endif()

if(TARGET ifpsolver)
    addDirectory(ALIEN/Kernels/IFP)
endif()

if(TARGET htssolver)
  addDirectory(ALIEN/Kernels/HTS)
endif()

if(TARGET arcane_core)
  linkLibraries(alien_ifpen_solvers arcane_core arcane_std)
else ()
    linkLibraries(alien_ifpen_solvers ALIEN-StrongOptions)
endif()

if(TARGET umfpack)
  linkLibraries(alien_ifpen_solvers umfpack)
endif()

linkLibraries(alien_ifpen_solvers alien)
linkLibraries(alien_ifpen_solvers Boost::system)

target_include_directories(alien_ifpen_solvers PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

commit(alien_ifpen_solvers)

install(DIRECTORY ALIEN
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
  )
