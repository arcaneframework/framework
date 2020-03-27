if (NOT TARGET hpddm)
    return()
endif ()

createLibrary(ALIEN-HPDDM
  EXPORT ALIEN/ALIEN-HPDDMExport.h
  )

set_target_properties(ALIEN-HPDDM
		      PROPERTIES LINKER_LANGUAGE "CXX")

if(TARGET arcane_core)
	set(AXL_OPTION_GENERATION_TYPE "ALL")
	set(NAMESPACE_TYPE "Arcane")
	set(IS_WITH_ARCANE "")
	set(IS_WITH_MESH "NO_MESH")
else()
	set(AXL_OPTION_GENERATION_TYPE "STRONG_OPTIONS_ONLY")
	set(NAMESPACE_TYPE "Alien")
	set(IS_WITH_ARCANE "NO_ARCANE")
	set(IS_WITH_MESH "NO_MESH")
endif()

if(TARGET hpddm)
  addDirectory(ALIEN/Kernels/HPDDM)
endif()

if(TARGET arcane_core)
  linkLibraries(ALIEN-HPDDM arcane_core arcane_std)
else ()
  linkLibraries(ALIEN-HPDDM ALIEN-StrongOptions)
endif()

linkLibraries(ALIEN-HPDDM alien)

target_include_directories(ALIEN-HPDDM PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

commit(ALIEN-HPDDM)

install(DIRECTORY ALIEN
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
  )
