if (NOT TARGET arcane_core)
    return()
endif ()

createLibrary(ALIEN-ArcaneTools
  EXPORT ALIEN/ALIEN-ArcaneToolsExport.h
  )

addSources(ALIEN-ArcaneTools
  ALIEN/ArcaneTools/Accessors/ItemVectorAccessor.cc
  #ALIEN/ArcaneTools/IndexManager/ArcGeoSim/ContactAbstractFamily.cc
  ALIEN/ArcaneTools/IndexManager/SimpleAbstractFamily.cc
  ALIEN/ArcaneTools/IndexManager/BasicIndexManager.cc
  ALIEN/ArcaneTools/IndexManager/AnyItemAbstractFamily.cc
  ALIEN/ArcaneTools/Block/IndexManagerBlockBuilder.cc
  )

if(TARGET petsc)
  addSources(ALIEN-ArcaneTools 
    ALIEN/ArcaneTools/IO/MatrixRowPrinter.cc
    )
  linkLibraries(ALIEN-ArcaneTools petsc)
  linkLibraries(ALIEN-ArcaneTools ALIEN-ExternalPackages)
endif()

linkLibraries(ALIEN-ArcaneTools alien)

commit(ALIEN-ArcaneTools)

target_include_directories(ALIEN-ArcaneTools PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY ALIEN 
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
  )
