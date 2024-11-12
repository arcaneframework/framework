set(PRIVATE_PKGS LibUnwind Papi Parmetis PTScotch Udunits Zoltan BZip2 LZ4 Otf2 DbgHelp HWLoc Hiredis)
set(PUBLIC_PKGS HDF5 MPI vtkIOXML vtkIOLegacy)
set(PKGS ${PRIVATE_PKGS} ${PUBLIC_PKGS})

foreach(package ${PKGS})
  arcane_find_package(${package})
endforeach()

# ----------------------------------------------------------------------------
# Pour l'instant n'active pas KDI car il n'y a pas d'installation disponibles
set(ARCANE_HAS_KDI FALSE)

# Nécessite CMake 3.26 pour la détection de l'API stable de Python
# NOTE: aout 2024. Cependant il semble que cette API stable ne soit pas
# toujours bien détectée (la cible CMake associée ne contient rien)
# On utilise donc la cible Python classique temporairement

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
  message(STATUS "Checking support for KDI Python")
  set(Python_USE_STATIC_LIBS FALSE)
  find_package (Python 3.10 COMPONENTS Development Development.SABIModule NumPy)
  if (Python_FOUND AND Python_Development.SABIModule_FOUND AND Python_NumPy_FOUND)
    set(ARCANE_HAS_KDI_PYTHON TRUE)
    message(STATUS "Python version = ${Python_VERSION}")
    message(STATUS "Python libs = ${Python_LIBRARIES} rtlibs=${Python_RUNTIME_LIBRARY_DIRS}")
    message(STATUS "Python SABI libs = ${Python_SABI_LIBRARY} ${Python_SABI_LIBRARIES} ${Python_RUNTIME_SABI_LIBRARY_DIRS} ${Python_SABI_LIBRARY_DIRS}")
    message(STATUS "Python NumPy version = ${Python_NumPy_VERSION}")
    message(STATUS "Python NumPy include_dirs = ${Python_NumPy_INCLUDE_DIRS}")
    if (Python_NumPy_VERSION VERSION_LESS 1.23)
      message(STATUS "Disabling Python_NumPy because version is too old (minimum=1.23)")
      set (ARCANE_HAS_KDI_PYTHON FALSE)
    endif()
  else()
    set (ARCANE_HAS_KDI_PYTHON FALSE)
  endif()
endif()

if (NOT ARCANE_HAS_KDI_PYTHON)
  set (ARCANE_HAS_KDI FALSE)
endif()

message(STATUS "Is KDI available? = ${ARCANE_HAS_KDI}")
set(ARCANE_HAS_KDI ${ARCANE_HAS_KDI} CACHE BOOL "Is KDI available" FORCE)

# ----------------------------------------------------------------------------

# Pour MEDFile, il faut HDF5 mais le find_package fait planter CMake (avec MED 4.0.0)
# si HDF5 n'est pas trouvé. On ne fait donc le find_package que si HDF5
# est trouvé
if(HDF5_FOUND)
  list(APPEND PRIVATE_PKGS MEDFile)
  arcane_find_package(MEDFile QUIET)
endif()

include(srcs.cmake)

if(BZIP2_FOUND)
  list(APPEND ARCANE_SOURCES Bzip2DeflateService.cc)
endif()
if(LZ4_FOUND)
  list(APPEND ARCANE_SOURCES LZ4DeflateService.cc)
endif()
if(HDF5_FOUND)
  list(APPEND ARCANE_SOURCES
    VtkHdfPostProcessor.cc
    VtkHdfV2PostProcessor.cc
    )
endif()

if(MEDFILE_FOUND)
  list(APPEND ARCANE_SOURCES MEDMeshReaderService.cc)
endif()
if(Parmetis_FOUND)
  list(APPEND ARCANE_SOURCES
    MetisMeshPartitioner.cc
    MetisGraphDigest.cc
    MetisGraphGather.cc
    MetisWrapper.cc
    internal/MetisGraphDigest.h
    internal/MetisGraphGather.h
    internal/MetisGraph.h
    internal/MetisWrapper.h

    )
endif()
if(ZOLTAN_FOUND)
  list(APPEND ARCANE_SOURCES ZoltanMeshPartitioner.cc)
endif()
if(PTSCOTCH_FOUND)
  list(APPEND ARCANE_SOURCES PTScotchMeshPartitioner.cc)
endif()
if(LIBUNWIND_FOUND)
  list(APPEND ARCANE_SOURCES LibUnwindStackTraceService.cc)
endif()
if(UDUNITS_FOUND)
  list(APPEND ARCANE_SOURCES UdunitsPhysicalUnitSystemService.cc)
endif()
if(PAPI_FOUND)
  list(APPEND ARCANE_SOURCES PapiPerformanceService.cc)
endif()
if(LIBUNWIND_FOUND)
  list(APPEND ARCANE_SOURCES ProfPerformanceService.cc)
endif()
if(HWLOC_FOUND)
  list(APPEND ARCANE_SOURCES HWLocProcessorAffinity.cc)
endif()
if(OTF2_FOUND)
  if(TARGET Arccore::arccore_message_passing_mpi)
    # Récupère le numéro de version de Otf2 et vérifie qu'il s'agit
    # de la version 2.1.
    # TODO: à terme mettre cette détection dans Arccon.
    find_file(_OTF_GEN_DEF NAMES otf2/OTF2_GeneralDefinitions.h HINTS ${OTF2_INCLUDE_DIRS})
    if(_OTF_GEN_DEF)
      set(OTF2_REQUIRED_VERSION "2.1")
      file(STRINGS ${_OTF_GEN_DEF} _OTF_GEN_DEF_CONTENT REGEX "OTF2_VERSION")
      string(REGEX MATCH "OTF2_VERSION_MAJOR[ ]+([0-9]+).*OTF2_VERSION_MINOR[ ]+([0-9]+)" OTF2_VERSION_MATCH ${_OTF_GEN_DEF_CONTENT})
      set(OTF2_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
      message(STATUS "[Otf2] Found version '${OTF2_VERSION}' (required=${OTF2_REQUIRED_VERSION})")
      if(OTF2_VERSION VERSION_LESS ${OTF2_REQUIRED_VERSION})
        message(FATAL_ERROR
          "[Otf2] Found version (${OTF2_VERSION}) of Otf2 is too old."
          " Version '${OTF2_REQUIRED_VERSION}+' is required."
          " You may disable 'Otf2' if you add '-DCMAKE_DISABLE_FIND_PACKAGE_Otf2=TRUE'"
          " to CMmake arguments during configuration"
          )
      endif()
    else()
      message(FATAL_ERROR "Can not determine OTF2 Version")
    endif()
    list(APPEND ARCANE_SOURCES
      Otf2MessagePassingProfilingService.cc
      Otf2MpiProfiling.cc
      Otf2LibWrapper.cc
      )
  endif()
endif()
if(DbgHelp_FOUND)
  list(APPEND ARCANE_SOURCES
    DbgHelpStackTraceService.cc
    )
endif()
if(ARCANE_HAS_LINUX_PERF_COUNTERS)
  list(APPEND ARCANE_SOURCES
    LinuxPerfPerformanceCounterService.cc
    )
endif()

if(vtkIOXML_FOUND AND vtkIOLegacy_FOUND)
  list(APPEND ARCANE_SOURCES
    VtkPolyhedralMeshIOService.cc
    )
endif()

if (ARCANE_HAS_KDI_PYTHON)
  list(APPEND ARCANE_SOURCES
    KdiPostProcessor.cc
    internal/Kdi.h
  )
endif()

# ----------------------------------------------------------------------------

arcane_add_library(arcane_std
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/std
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
  )

# ----------------------------------------------------------------------------

if(${ARCANE_HAS_MALLOC_HOOKS})
  target_compile_definitions(arcane_std PRIVATE ARCANE_USE_MALLOC_HOOK)
endif()

if (ARCANE_HAS_KDI_PYTHON)
  target_link_libraries(arcane_std PRIVATE Python::Python Python::NumPy)
endif()

target_link_libraries(arcane_std PUBLIC $<TARGET_NAME_IF_EXISTS:Arccore::arccore_message_passing_mpi>)
target_link_libraries(arcane_std PUBLIC $<TARGET_NAME_IF_EXISTS:Arcane::arcane_hdf5>)
target_link_libraries(arcane_std PUBLIC ${ARCANE_BASE_LIBRARIES})

# TODO: il semblerait qu'on ait besoin de rajouter stdc++fs pour que link passe avec le
# std::filesystem du c++17 pour le moment, a enlever quand ce sera possible
if(OTF2_FOUND AND UNIX)
  target_link_libraries(arcane_std PUBLIC stdc++fs)
endif()

arcane_add_arccon_packages(arcane_std PRIVATE ${PRIVATE_PKGS})
arcane_add_arccon_packages(arcane_std PUBLIC ${PUBLIC_PKGS})

arcane_register_library(arcane_std)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
