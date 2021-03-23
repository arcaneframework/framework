set(PKGS LibUnwind Papi Parmetis PTScotch Udunits Zoltan HDF5 MPI MKL BZip2 LZ4 Otf2 DbgHelp)
# HWLoc est pour l'instant incompatible avec MPC (probablement pas completement thread-safe)
# NOTE: Cela n'a pas été testé depuis 2011.
if(NOT ARCANE_USE_MPC)
  list(APPEND PKGS HWLoc)
endif()

foreach(package ${PKGS})
  find_package(${package})
endforeach()

# Pour MEDFile, il faut HDF5 mais le find_package fait planter CMake (avec MED 4.0.0)
# si HDF5 n'est pas trouvé. On ne fait donc le find_package que si HDF5
# est trouvé
if(HDF5_FOUND)
  list(APPEND PKGS MEDFile)
  find_package(MEDFile QUIET)
endif()

include(srcs.cmake)

if(BZIP2_FOUND)
  set(ARCANE_SOURCES ${ARCANE_SOURCES} Bzip2DeflateService.cc)
endif()
if(LZ4_FOUND)
  set(ARCANE_SOURCES ${ARCANE_SOURCES} LZ4DeflateService.cc)
endif()
if(HDF5_FOUND)
  set(ARCANE_SOURCES ${ARCANE_SOURCES}
    Hdf5Utils.cc
    Hdf5VariableInfoBase.cc
    Hdf5ReaderWriter.cc
    Hdf5VariableReader.cc
    Hdf5VariableWriter.cc
    Hdf5MpiReaderWriter.cc
    EnsightHdfPostProcessor.cc
    )
endif()

if(MEDFILE_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} MEDMeshReaderService.cc)
endif()
if (Parmetis_FOUND)
  list(APPEND ARCANE_SOURCES
    MetisMeshPartitioner.cc
    MetisGraphDigest.cc
    MetisGraphGather.cc
    MetisWrapper.cc
    MetisGraph.h
    MetisGraphDigest.h
    MetisGraphGather.h
    MetisWrapper.h
)
endif()
if(ZOLTAN_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} ZoltanMeshPartitioner.cc)
endif()
if(PTSCOTCH_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} PTScotchMeshPartitioner.cc)
endif()
if(LIBUNWIND_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} LibUnwindStackTraceService.cc)
endif()
if(UDUNITS_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} UdunitsPhysicalUnitSystemService.cc)
endif()
if(PAPI_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} PapiPerformanceService.cc)
endif()
if(PERFMON_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} PerfmonPerformanceService.cc)
endif()
if(LIBUNWIND_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} ProfPerformanceService.cc)
endif()
if(HWLOC_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} HWLocProcessorAffinity.cc)
endif()
if(MKL_FOUND)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} MKLVectorMathService.cc)
endif()
if(OTF2_FOUND)
  if (TARGET Arccore::arccore_message_passing_mpi)
    set( ARCANE_SOURCES ${ARCANE_SOURCES}
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

arcane_add_library(arcane_std
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/std
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

# ----------------------------------------------------------------------------

if (TARGET Arccore::arccore_message_passing_mpi)
  target_link_libraries(arcane_std PUBLIC Arccore::arccore_message_passing_mpi)
endif()
target_link_libraries(arcane_std PUBLIC ${ARCANE_BASE_LIBRARIES})

# TODO: il semblerait qu'on ait besoin de rajouter stdc++fs pour que link passe avec le
# std::filesystem du c++17 pour le moment, a enlever quand ce sera possible
if (OTF2_FOUND AND UNIX)
  target_link_libraries(arcane_std PUBLIC stdc++fs)
endif()

arcane_add_arccon_packages(arcane_std PRIVATE ${PKGS})

arcane_register_library(arcane_std)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
