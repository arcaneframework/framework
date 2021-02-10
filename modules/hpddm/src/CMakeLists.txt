if (NOT TARGET hpddm)
    return()
endif ()

createLibrary(alien_hpddm
  EXPORT alien/AlienHPDDMExport.h
  )

set_target_properties(alien_hpddm
		      PROPERTIES LINKER_LANGUAGE "CXX")

if(TARGET arcane_core)
	set(AXL_OPTION_GENERATION_TYPE "ALL")
	set(NAMESPACE_TYPE "Arcane")
	set(IS_WITH_ARCANE "")
	set(IS_WITH_MESH "NO_MESH")
else()
	set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
	set(NAMESPACE_TYPE "Arccore")
	set(IS_WITH_ARCANE "NO_ARCANE")
	set(IS_WITH_MESH "NO_MESH")
endif()

if(TARGET hpddm)
  addDirectory(alien/kernels/hpddm)
endif()

if(TARGET arcane_core)
  linkLibraries(alien_hpddm arcane_core arcane_std)
else ()
  linkLibraries(alien_hpddm alien_strong_options)
endif()

linkLibraries(alien_hpddm Alien::alien_core Alien::alien_semantic_ref)

target_include_directories(alien_hpddm PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

commit(alien_hpddm)

install(DIRECTORY alien
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
  )
