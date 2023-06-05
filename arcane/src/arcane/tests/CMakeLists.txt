set(TEST_DIRS . anyitem dof inout)

set(TEST_LIBS ${ARCANE_ADDITIONNAL_TEST_LIBRARIES})
if(MPI_FOUND AND NOT WIN32)
  # Aleph n'est disponible que si on a MPI et Linux
  set(TEST_LIBS ${TEST_LIBS} arcane_aleph_tests)
endif()
set(TEST_LIBS ${TEST_LIBS} ${ARCANE_LIBRARIES})

if(ARCANE_HAS_SOLVERS)
  set(TEST_DIRS ${TEST_DIRS} solvers)
  set(TEST_LIBS ${TEST_LIBS} arcane_solvers)
endif()

if(ARCANE_HAS_GEOMETRY)
  set(TEST_DIRS ${TEST_DIRS} geometry)
  set(TEST_LIBS ${TEST_LIBS} arcane_geometry)
endif()

if(GEOMETRYKERNEL_FOUND)
  set(TEST_DIRS ${TEST_DIRS} corefinement)
  set(TEST_LIBS ${TEST_LIBS} arcane_corefinement)
endif()
set(PKGS TBB)
arcane_find_package(TBB)

arcane_find_package(Neo QUIET)
if (Neo_FOUND)
  set(TEST_LIBS ${TEST_LIBS} Neo::Neo)
endif()

arcane_find_package(hiredis QUIET)

include(srcs.cmake)

set(ARCANE_ACCELERATOR_SOURCES
  accelerator/SimpleHydroAcceleratorService.cc
  accelerator/MiniWeatherArray.cc
  accelerator/MemoryCopyUnitTest.cc
  accelerator/MultiMemoryCopyUnitTest.cc
  accelerator/NumArrayUnitTest.cc
  accelerator/AcceleratorReduceUnitTest.cc
  accelerator/RunQueueUnitTest.cc
  accelerator/AcceleratorViewsUnitTest.cc
  accelerator/ArcaneTestStandaloneAcceleratorMng.cc
  accelerator/MeshMaterialAcceleratorUnitTest.cc
  )
list(APPEND ARCANE_SOURCES ${ARCANE_ACCELERATOR_SOURCES})
if (TARGET hiredis::hiredis)
  list(APPEND ARCANE_SOURCES RedisUnitTest.cc)
endif()

arcane_accelerator_add_source_files(${ARCANE_ACCELERATOR_SOURCES})

set(CURRENT_SRC_PATH ${Arcane_SOURCE_DIR}/src)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(BINARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

arcane_add_library(arcane_tests_lib
  INPUT_PATH ${CURRENT_SRC_PATH}
  RELATIVE_PATH arcane/tests
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
  )

arcane_add_arccon_packages(arcane_tests_lib PRIVATE ${PKGS})

add_executable(arcane_tests_exec ${ARCANE_SRC_PATH}/arcane/tests/ArcaneTestMain.cc)
set(ARCANE_TEST_PATH ${ARCANEBUILDROOT}/src/arcane/tests)

# Recopie dans le répertoire les DLLs nécessaires à l'exécutable de test
if (WIN32)
  add_custom_command(TARGET arcane_tests_exec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:arcane_tests_exec> $<TARGET_FILE_DIR:arcane_tests_exec>
    COMMAND_EXPAND_LISTS
    )
endif()

file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests/geometry)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests/solvers)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests/anyitem)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests/inout)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests/dof)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests/discreteoperator)
file(MAKE_DIRECTORY ${ARCANEBUILDROOT}/arcane/tests/corefinement)
file(MAKE_DIRECTORY ${ARCANE_TEST_PATH})

target_link_libraries(arcane_tests_exec PRIVATE ${ADD_NO_AS_NEEDED} arcane_tests_lib)
target_link_libraries(arcane_tests_lib PUBLIC ${ADD_NO_AS_NEEDED} ${TEST_LIBS} arcane_full ${ARCANE_LIBRARIES})

# Pour mettre l'executable dans 'lib'
arcane_target_set_standard_path(arcane_tests_exec)
#set_target_properties(arcane_tests_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set_target_properties(arcane_tests_lib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

configure_file(ArcaneTest.config ${CMAKE_BINARY_DIR}/lib COPYONLY)

if (Neo_FOUND)
  target_compile_definitions(arcane_tests_exec PRIVATE ARCANE_HAS_CUSTOM_MESH_TOOLS)
  target_compile_definitions(arcane_tests_lib PUBLIC ARCANE_HAS_CUSTOM_MESH_TOOLS)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion des tests C#.
# Il faut le wrapper C# pour pouvoir utiliser ces tests
if (TARGET arcane_dotnet_wrapper_services)
  set(ARCANE_HAS_DOTNET_TESTS TRUE)
  message(STATUS "** ** USING C# TEST LIBS=${ARCANE_LIBRARIES}")
  set(OUTLIB ${LIBRARY_OUTPUT_PATH}/ArcaneTest.dll)
  set(CSPATH ${ARCANE_SRC_PATH}/arcane/tests)
  set(CSOUTPATH ${ARCANEBUILDROOT}/arcane/tests)
  configure_file(ArcaneTest.csproj.in ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneTest/ArcaneTest.csproj @ONLY)

  add_custom_command(OUTPUT ${CSOUTPATH}/SimpleHydroCS_axl.cs
    DEPENDS ${CSPATH}/SimpleHydroCS.axl ${ARCANE_AXL2CC}
    COMMAND ${ARCANE_AXL2CC}
    ARGS -i arcane/tests/. --lang c\# -o ${CSOUTPATH} ${CSPATH}/SimpleHydroCS.axl)

  add_custom_command(OUTPUT ${CSOUTPATH}/PerfectGasEOS_axl.cs
    DEPENDS ${CSPATH}/PerfectGasEOS.axl ${ARCANE_AXL2CC}
    COMMAND ${ARCANE_AXL2CC}
    ARGS -i arcane/tests/. --lang c\# -o ${CSOUTPATH} ${CSPATH}/PerfectGasEOS.axl)

  add_custom_command(OUTPUT ${CSOUTPATH}/MeshModification_axl.cs
    DEPENDS ${CSPATH}/MeshModification.axl ${ARCANE_AXL2CC}
    COMMAND ${ARCANE_AXL2CC}
    ARGS -i arcane/tests/. --lang c\# -o ${CSOUTPATH} ${CSPATH}/MeshModification.axl)

  arccon_add_csharp_target(arcane_test_cs
    DOTNET_RUNTIME ${ARCANE_DOTNET_RUNTIME}
    BUILD_DIR ${LIBRARY_OUTPUT_PATH}
    ASSEMBLY_NAME ArcaneTest.dll
    PROJECT_PATH ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneTest
    PROJECT_NAME ArcaneTest.csproj
    MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS}
    DEPENDS
    ${CSOUTPATH}/SimpleHydroCS_axl.cs
    ${CSOUTPATH}/PerfectGasEOS_axl.cs
    ${CSOUTPATH}/MeshModification_axl.cs
    DOTNET_TARGET_DEPENDS dotnet_wrapper_services dotnet_wrapper_launcher
    )

  arccon_add_csharp_target(arcane_test_cs_exe
    DOTNET_RUNTIME ${ARCANE_DOTNET_RUNTIME}
    BUILD_DIR ${LIBRARY_OUTPUT_PATH}
    ASSEMBLY_NAME ArcaneTestExe.dll
    PROJECT_PATH ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneTestExe
    PROJECT_NAME ArcaneTestExe.csproj
    MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS}
    DOTNET_TARGET_DEPENDS arcane_test_cs
    )

  set(OUTEXE ${LIBRARY_OUTPUT_PATH}/ArcaneTestExe.dll)
  configure_file(ArcaneTestExe.csproj.in ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneTestExe/ArcaneTestExe.csproj @ONLY)

  #add_custom_target(arcane_tests_cs_all ALL DEPENDS ${OUTEXE} ${OUTLIB})
  # Pour que la cible soit automatiquement prise en compte dans les tests
  add_dependencies(arcane_tests_exec arcane_test_cs_exe)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(TEST_PATH ${ARCANESRCROOT}/tests)
if(WIN32)
  set(ARCANE_TEST_DRIVER ${ARCANEBUILDROOT}/bin/arcane_test_driver.bat)
  set(ARCANE_CONVERT_MESH_COMMAND ${ARCANEBUILDROOT}/bin/arcane_convert_mesh.bat)
else()
  set(ARCANE_TEST_DRIVER ${ARCANEBUILDROOT}/bin/arcane_test_driver)
  set(ARCANE_CONVERT_MESH_COMMAND ${ARCANEBUILDROOT}/bin/arcane_convert_mesh)
endif()
# Sauve cette variable dans le cache pour que des autres projets utilisant Arcane puissent l'utiliser.
set(ARCANE_TEST_DRIVER ${ARCANE_TEST_DRIVER} CACHE STRING "Driver for test" FORCE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

target_link_libraries(arcane_tests_lib PRIVATE arcane_accelerator)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(ARCANE_TEST_WORKDIR ${CMAKE_BINARY_DIR}/src/arcane/tests)
set(ARCANE_TEST_CASEPATH ${TEST_PATH})
set(ARCANE_TEST_EXECNAME arcane_tests_exec)
set(ARCANE_TEST_DOTNET_ASSEMBLY ${LIBRARY_OUTPUT_PATH}/ArcaneTest.dll)
include(ArcaneTests.cmake)

# ----------------------------------------------------------------------------

function(ARCANE_COPY_MESH orig_mesh_name mesh_name ext)
  exec_program(${CMAKE_COMMAND} ARGS -E copy_if_different ${ARCANESRCROOT}/maillages/${orig_mesh_name}.${ext}
    ${ARCANE_TEST_PATH}/${mesh_name}.${ext})
  exec_program(${CMAKE_COMMAND} ARGS -E copy_if_different ${ARCANESRCROOT}/maillages/${orig_mesh_name}.vtkfaces.${ext}
    ${ARCANE_TEST_PATH}/${mesh_name}.${ext}faces.${ext})
endfunction()
function(ARCANE_COPY_MESH_DIRECT orig_mesh_name)
  exec_program(${CMAKE_COMMAND} ARGS -E copy_if_different ${ARCANESRCROOT}/maillages/${orig_mesh_name}
    ${ARCANE_TEST_PATH}/${orig_mesh_name})
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Copie les maillages de test
MESSAGE(STATUS "Copying data for tests")
ARCANE_COPY_MESH(tube5x5x100 sod vtk)
ARCANE_COPY_MESH(tube5x5x100 tube5x5x100 vtk)
ARCANE_COPY_MESH(sphere sphere vtk)
ARCANE_COPY_MESH(plan plan vtk)
ARCANE_COPY_MESH_DIRECT(voronoi.vor)
ARCANE_COPY_MESH_DIRECT(voronoi.format)
ARCANE_COPY_MESH(quadratic3d quadratic3d vtk)
ARCANE_COPY_MESH_DIRECT(elements.vtu)
ARCANE_COPY_MESH_DIRECT(aomd.msh)
ARCANE_COPY_MESH_DIRECT(square_v41.msh)
ARCANE_COPY_MESH_DIRECT(xml_data.xmf)
ARCANE_COPY_MESH_DIRECT(mesh.h5)
ARCANE_COPY_MESH_DIRECT(merge_nodes_2d.vtk)
ARCANE_COPY_MESH_DIRECT(mesh1.med)
ARCANE_COPY_MESH_DIRECT(mesh2.med)
arcane_copy_mesh_direct(tube5x5x100.mli)
arcane_copy_mesh_direct(tube5x5x100.unf)
arcane_copy_mesh_direct(sod3d-penta6.msh)
arcane_copy_mesh_direct(sod3d-misc.msh)
arcane_copy_mesh_direct(planar_unstructured_quad1.msh)
arcane_copy_mesh_direct(faultx1_2x1x1.vtk)
arcane_copy_mesh_direct(faultx1_2x1x1.vtkfaces.vtk)
arcane_copy_mesh_direct(plancher.msh)
arcane_copy_mesh(tied_interface_1 tied_interface_1 vtk)
arcane_copy_mesh(tied_interface_2 tied_interface_2 vtk)
arcane_copy_mesh(tied_interface_2d_1 tied_interface_2d_1 vtk)
arcane_copy_mesh(sphere_tied_1 sphere_tied_1 vtk)
arcane_copy_mesh(sphere_tied_2 sphere_tied_2 vtk)
arcane_copy_mesh(tied_interface_2d_1_v4.2 tied_interface_2d_1_v4.2 vtk)
arcane_copy_mesh(sphere_v4.2 sphere_v4.2 vtk)
arcane_copy_mesh(sphere_v4.2_binary sphere_v4.2_binary vtk)

configure_file(test-CasePartVtu.sh.in ${ARCANE_TEST_PATH}/test-CasePartVtu.sh)

# Copie le répertoire des données dans le répertoire d'exécution des tests
# Cela est nécessaire pour certains cas comme ceux faisant une reprise
# depuis une ancienne version de Arcane

# La protection pour ce test est issue du test hydro3_cartesian_small
file(COPY ${TEST_PATH}/data/test_output_continue_hydro3_cartesian_small
  DESTINATION ${ARCANE_TEST_WORKDIR} USE_SOURCE_PERMISSIONS)
# La protection pour ce test est issue du test hydro3_cartesian_small_4thread
file(COPY ${TEST_PATH}/data/test_output_continue_hydro3_cartesian_small_4shm
  DESTINATION ${ARCANE_TEST_WORKDIR} USE_SOURCE_PERMISSIONS)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

message(STATUS "ADD AUTOMATED TESTS")
ARCANE_ADD_TEST_SEQUENTIAL(utils1 testUtils-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(json1 testJSON-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(xml1 testXml-1.arc -We,ARCANE_XML_PATH,${ARCANE_ARC_PATH})
ARCANE_ADD_TEST_SEQUENTIAL(random1 testRandom-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(simd1 testSimd-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(singleton1 testSingletonService-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(array1 testArcane-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(configuration1 testConfiguration-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(properties1 testProperties-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(itemvector1 testItemVector-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(variable testVariable-1.arc)
if (LZ4_FOUND)
  ARCANE_ADD_TEST_SEQUENTIAL(variable_lz4 testVariable-1-lz4.arc)
endif()
ARCANE_ADD_TEST_SEQUENTIAL(mdvariable testMDVariable-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(variablesimd testVariableSimd-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(partial-variable-seq testPartialVariable-seq.arc)
if(PARTITIONER_FOUND)
  ARCANE_ADD_TEST_PARALLEL(partial-variable testPartialVariable.arc 4)
endif()
ARCANE_ADD_TEST_SEQUENTIAL(caseoptions testCaseOptions-1.arc)
arcane_add_test_sequential(caseoptions_specificentrypoint testCaseOptions-specific-entry-point.arc
  "-We,ARCANE_CALL_SPECIFIC_ENTRY_POINT,CaseOptionLoop2")
if(UDUNITS_FOUND)
  ARCANE_ADD_TEST_SEQUENTIAL(caseoptions_unit testCaseOptions-2.arc)
endif()
ARCANE_ADD_TEST_SEQUENTIAL(caseoptions_opt testCaseOptions-3.arc)
ARCANE_ADD_TEST_SEQUENTIAL(caseoptions_default testCaseOptions-4.arc)
ARCANE_ADD_TEST_SEQUENTIAL(caseoptions_category testCaseOptions-5.arc -We,ARCANE_DEFAULT_CATEGORY,X1)
arcane_add_test_sequential(caseoptions_multiplemesh testCaseOptions-multiple-mesh.arc -We,ARCANE_DEFAULT_CATEGORY,X3)
arcane_add_test_sequential(caseoptions_multiplemesh_legacy testCaseOptions-multiple-mesh-legacy.arc -We,ARCANE_DEFAULT_CATEGORY,X3)
arcane_add_test_sequential(casefunction1 testCaseFunction-1.arc)
if (hiredis_FOUND)
  # Pour ce test on a aussi besoin qu'une base redis soit accessible
  # Comme il n'y a pas de moyen simple de le savoir, ce test n'est pas actif
  # par défaut.
  # arcane_add_test_sequential(redis1 testRedis-1.arc)
endif()

ARCANE_ADD_TEST(timehistory testTimeHistory-1.arc)

add_test(
  NAME direct_exec1
  COMMAND ${ARCANE_TEST_DRIVER} launch -D "TestRunDirect1"
  WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR}
  )
# Test en spécifiant le JDD avec le paramètre 'CaseDatasetFileName'.
add_test(
  NAME direct_exec1_case_arg
  COMMAND ${ARCANE_TEST_DRIVER} launch "-A,CaseDatasetFileName=${ARCANE_TEST_CASEPATH}/testRunDirect-1.arc" -D TestRunDirectCartesian
  WORKING_DIRECTORY ${ARCANE_TEST_WORKDIR}
  )
arcane_add_test_sequential(direct_exec_cartesian_sequential testRunDirect-1.arc -D TestRunDirectCartesianSequential)
arcane_add_test(direct_exec_hello_world testRunDirect-1.arc -D TestRunDirectHelloWorld)
arcane_add_test(direct_exec_cartesian testRunDirect-1.arc -D TestRunDirectCartesian -We,ARCANE_USE_OLD_MESHGENERATOR,1)

ADD_TEST(dump_all_internal ${ARCANE_TEST_DRIVER} launch -We,STDENV_CODE_NAME,Arcane -arcane_opt arcane_all_internal)
ADD_TEST(dump_internal ${ARCANE_TEST_DRIVER} launch -We,STDENV_CODE_NAME,Arcane -arcane_opt arcane_internal)
ADD_TEST(dump_database ${ARCANE_TEST_DRIVER} launch -We,STDENV_CODE_NAME,Arcane -arcane_opt arcane_database)

# ----------------------------------------------------------------------------

arcane_add_test_sequential(standalone_accelerator_testsum dummy.arc "-A,StandaloneAcceleratorMethod=TestSum")
arcane_add_accelerator_test_sequential(standalone_accelerator_testsum dummy.arc "-A,StandaloneAcceleratorMethod=TestSum")
arcane_add_accelerator_test_sequential(standalone_accelerator_testemptykernel dummy.arc "-A,StandaloneAcceleratorMethod=TestEmptyKernel")

arcane_add_test_sequential(standalone_subdomain1 dummy.arc "-A,StandaloneSubDomainMethod=Test1")
arcane_add_test_sequential(standalone_subdomain2 dummy.arc "-A,StandaloneSubDomainMethod=Test2")
arcane_add_test_parallel(standalone_subdomain2 dummy.arc 4 "-A,StandaloneSubDomainMethod=Test2")

# ----------------------------------------------------------------------------

# Macro pour tester le message_passing
macro(add_test_message_passing test_name)
  set(_OPTS -We,MESSAGE_PASSING_TEST,${test_name} -arcane_opt direct_test ParallelMngTest null)
  set(_TEST_NAME parallelmng_${test_name})
  add_test(${_TEST_NAME}_seq ${ARCANE_TEST_DRIVER} launch ${_OPTS})
  if(TARGET arcane_mpi)
    add_test(${_TEST_NAME}_4proc ${ARCANE_TEST_DRIVER} launch -n 4 ${_OPTS})
  endif()
  add_test(${_TEST_NAME}_4thread ${ARCANE_TEST_DRIVER} launch -T 4 ${_OPTS})
  arcane_add_test_message_passing_hybrid(${_TEST_NAME} NB_MPI 3 NB_SHM 4 ARGS ${_OPTS})
endmacro()

add_test_message_passing(serialize)
add_test_message_passing(serialize_message_list)
add_test_message_passing(serializer_with_message_info)
add_test_message_passing(datatype)
add_test_message_passing(barrier)
add_test_message_passing(named_barrier)
add_test_message_passing(process_messages)
add_test_message_passing(send_receive_nb3)
add_test_message_passing(reduce2)
add_test_message_passing(topology)
add_test_message_passing(broadcast_serializer)
add_test_message_passing(all)
add_test_message_passing(sub_all)

# Pour la converture de test, lance les tests en mode hybrid mais avec uniquement 4 coeurs
arcane_add_test_message_passing_hybrid(parallelmng_all NB_MPI 2 NB_SHM 2 ARGS -We,MESSAGE_PASSING_TEST,all -arcane_opt direct_test ParallelMngTest null)
arcane_add_test_message_passing_hybrid(parallelmng_sub_all NB_MPI 2 NB_SHM 2 ARGS -We,MESSAGE_PASSING_TEST,all -arcane_opt direct_test ParallelMngTest null)

add_test(serialize_message_list_v2_4proc
  ${ARCANE_TEST_DRIVER} launch -n 4
  -We,MESSAGE_PASSING_TEST,serialize_message_list
  -We,ARCANE_SYNCHRONIZE_LIST_VERSION,2
  -arcane_opt direct_test ParallelMngTest null)

# ----------------------------------------------------------------------------

ARCANE_ADD_TEST_PARALLEL(parallel testParallel-1.arc 4)
ARCANE_ADD_TEST_PARALLEL(parallel2 testParallel-2.arc 4)
arcane_add_test_parallel(parallel2_synchronize testParallel-synchronize1.arc 4)
arcane_add_test_parallel_thread(parallel2_synchronize testParallel-synchronize1.arc 4)
arcane_add_test_parallel(parallel2_synchronize_legacymulti testParallel-synchronize1.arc 4 -We,ARCANE_USE_LEGACY_MULTISYNCHRONIZE,1)
arcane_add_test_parallel(parallel2_synchronize_v1 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,1)
arcane_add_test_parallel(parallel2_synchronize_v2 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,2)
arcane_add_test_parallel(parallel2_synchronize_v3 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,3)
arcane_add_test_parallel(parallel2_synchronize_v4 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,4 -We,ARCANE_SYNCHRONIZE_NB_SEQUENCE,3)
arcane_add_test_parallel(parallel2_synchronize_v4_b1024 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,4 -We,ARCANE_SYNCHRONIZE_BLOCK_SIZE,1024)
arcane_add_test_parallel(parallel2_synchronize testParallel-synchronize2.arc 8)
arcane_add_test_parallel(parallel2_synchronize_v1 testParallel-synchronize2.arc 8 -We,ARCANE_SYNCHRONIZE_VERSION,1)
arcane_add_test_parallel(parallel2_synchronize_v2 testParallel-synchronize2.arc 8 -We,ARCANE_SYNCHRONIZE_VERSION,2)
arcane_add_test_parallel(parallel2_synchronize_v3 testParallel-synchronize2.arc 8 -We,ARCANE_SYNCHRONIZE_VERSION,3)
arcane_add_test_parallel(parallel2_synchronize_v4 testParallel-synchronize2.arc 8 -We,ARCANE_SYNCHRONIZE_VERSION,4 -We,ARCANE_SYNCHRONIZE_NB_SEQUENCE,5)
arcane_add_test_parallel(parallel2_synchronize_v4_b1024 testParallel-synchronize2.arc 8 -We,ARCANE_SYNCHRONIZE_VERSION,4 -We,ARCANE_SYNCHRONIZE_BLOCK_SIZE,1024)
if (ARCANE_HAS_MPI_NEIGHBOR)
  arcane_add_test_parallel(parallel2_synchronize_v5 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,5)
  arcane_add_test_parallel(parallel2_synchronize_v5 testParallel-synchronize2.arc 8 -We,ARCANE_SYNCHRONIZE_VERSION,5)
endif()
ARCANE_ADD_TEST_PARALLEL(parallel2_mpiprof_json testParallel-2.arc 4 -We,ARCANE_MESSAGE_PASSING_PROFILING,JSON)
if(Otf2_FOUND)
  ARCANE_ADD_TEST_PARALLEL(parallel2_mpiprof_otf2 testParallel-2.arc 4 -We,ARCANE_MESSAGE_PASSING_PROFILING,OTF2)
endif()
ARCANE_ADD_TEST_PARALLEL(parallel2_2pe_rep3 testParallel-2.arc 6 -R 3)
ARCANE_ADD_TEST_PARALLEL_THREAD(parallel2_thread testParallel-2.arc 4)
ARCANE_ADD_TEST_PARALLEL(cell-submesh testSubMesh.arc 1)
ARCANE_ADD_TEST_PARALLEL(cell-submesh testSubMesh.arc 12)
ARCANE_ADD_TEST_PARALLEL(face-submesh testSubMesh2.arc 1)
ARCANE_ADD_TEST_PARALLEL(face-submesh testSubMesh2.arc 12)
arcane_add_test_parallel_all(face-submesh_2d testSubMesh2-2D.arc 2 2)
if(PARTITIONER_FOUND)
  ARCANE_ADD_TEST_PARALLEL(loadbalance testLoadBalance-1.arc 4)
  ARCANE_ADD_TEST_PARALLEL_THREAD(loadbalance testParticle.arc 4)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_rep3 testLoadBalance-1.arc 12 -R 3)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_collective testLoadBalance-1.arc 4 -We,ARCANE_MESH_EXCHANGE_USE_COLLECTIVE,1)
endif()
arcane_add_test_parallel_all(particle testParticle.arc 3 4)
arcane_add_test_parallel_all(particle_nonblocking testParticleNonBlocking.arc 3 4)
arcane_add_test_sequential(particle_async testParticleAsync.arc)
arcane_add_test_parallel(particle_async testParticleAsync.arc 4)
ARCANE_ADD_TEST_SEQUENTIAL(voronoi testVoronoi.arc -We,ARCANE_ITEM_TYPE_FILE,voronoi.format)
ARCANE_ADD_TEST_PARALLEL(voronoi testVoronoi.arc 4 -We,ARCANE_ITEM_TYPE_FILE,voronoi.format)
arcane_add_test(mesh testMesh-1.arc)
arcane_add_test_parallel_all(mesh_service testMeshService-1.arc 3 4)
ARCANE_ADD_TEST(mesh_2d testMesh-3.arc)
ARCANE_ADD_TEST_SEQUENTIAL(mesh_1d testMesh-4.arc)
arcane_add_test(multiple_mesh testMultipleMesh-1.arc -We,ARCANE_DUMP_VARIABLE_SYNCHRONIZER_TOPOLOGY,1)
arcane_add_test_sequential(dof testDoF.arc)
arcane_add_test_parallel(dof testDoF.arc 3)
arcane_add_test_sequential(dofnode testDoFNode.arc)
arcane_add_test_parallel(dofnode testDoFNode.arc 4)
ARCANE_ADD_TEST_SEQUENTIAL(graphdof_onecell testGraph-OneCell.arc -We,ARCANE_GRAPH_CONNECTIVITY_POLICY,1)
ARCANE_ADD_TEST_SEQUENTIAL(graphdof_twocells testGraph-TwoCells.arc -We,ARCANE_GRAPH_CONNECTIVITY_POLICY,1)
ARCANE_ADD_TEST_SEQUENTIAL(inout testInOut.arc)
ARCANE_ADD_TEST_SEQUENTIAL(mesh_merge_nodes_2d testMeshMergeNodes-2D.arc)
arcane_add_test_sequential(mesh_merge_boundaries testMeshMergeBoundaries-1.arc)
arcane_add_test_parallel(mesh_merge_boundaries testMeshMergeBoundaries-1.arc 4)
arcane_add_test_sequential(mesh_merge_boundaries2 testMeshMergeBoundaries-2.arc)
arcane_add_test_parallel(mesh_merge_boundaries2 testMeshMergeBoundaries-2.arc 4)
ARCANE_ADD_TEST_SEQUENTIAL(directed_graph testDirectedGraph.arc)
ARCANE_ADD_TEST_PARALLEL(directed_graph testDirectedGraph.arc 3)
arcane_add_test_sequential(mesh_deallocate testMeshDeallocate.arc)
arcane_add_test_parallel(mesh_deallocate testMeshDeallocate.arc 4)
if (ARCANE_DEFAULT_PARTITIONER_IS_METIS)
  arcane_add_test_parallel_thread(mesh_deallocate testMeshDeallocate.arc 4)
endif()
if(PARTITIONER_FOUND)
  ARCANE_ADD_TEST_PARALLEL(anyitem testAnyItem.arc 4)
  # Pour l'instant supprime le test de fusion car il ne marche pas en parallèle
  # (Il faudrait pour cela ajouter des contraintes au partitionnemenet)
  #  ARCANE_ADD_TEST_PARALLEL(mesh_merge_nodes_2d testMeshMergeNodes-2D.arc 4)
endif()

ARCANE_ADD_TEST(mesh2 testMesh-2.arc)
ARCANE_ADD_TEST_PARALLEL(mesh2_5ghost testMesh-2-5ghost.arc 4)
ARCANE_ADD_TEST_SEQUENTIAL(matvec testMatVec-1.arc)
#ARCANE_ADD_TEST_SEQUENTIAL(amr2 testAMR-2.arc)
arcane_add_test(amr1_2d testAMR-2D-1.arc)
if(HDF5_FOUND)
  ARCANE_ADD_TEST(checkpoint testCheckpoint-1.arc -c 3 -m 5)
  ARCANE_ADD_TEST(checkpoint_hdf5_g1 testCheckpoint-2.arc -c 3 -m 5)
  ARCANE_ADD_TEST(checkpoint_hdf5_g2 testCheckpoint-4.arc -c 3 -m 5)
  ARCANE_ADD_TEST(checkpoint_hdf5_g4 testCheckpoint-5.arc -c 3 -m 5)
endif()
if(vtkIOXML_FOUND)
  ARCANE_ADD_TEST_SEQUENTIAL(vtk_vtu testMesh-format_vtu.arc)
endif()
if(MEDFile_FOUND)
  arcane_add_test_sequential(mesh1_med testMesh-med1.arc)
  arcane_add_test(mesh2_med testMesh-med2.arc)
  arcane_add_test(mesh2_med_meshservice testMesh-med2-meshservice.arc)
endif()
ARCANE_ADD_TEST(mesh_quadratic3d_vtk testMesh-quadratic3d-vtk.arc)
ARCANE_ADD_TEST(mesh_tied_interface_2d_1_vtk42 testMesh-tied_interface_2d_1-vtk42.arc)
ARCANE_ADD_TEST(mesh_sphere_vtk42 testMesh-sphere-vtk42.arc)
ARCANE_ADD_TEST(mesh_sphere_vtk42_binary testMesh-sphere-vtk42-binary.arc)
arcane_add_test_sequential(mesh1_honeycomb2d testMesh-honeycomb2D-1.arc)
arcane_add_test_sequential(mesh1_honeycomb3d testMesh-honeycomb3D-1.arc)
if (ARCANE_DEFAULT_PARTITIONER_IS_METIS)
  arcane_add_test_parallel_thread(mesh1_honeycomb2d testMesh-honeycomb2D-1.arc 4)
  arcane_add_test_parallel_thread(mesh1_honeycomb3d testMesh-honeycomb3D-1.arc 4)
endif()

##############
# UNIT TESTS #
##############

ARCANE_ADD_TEST(unittest testUnitTest.arc)
# Ajoute un test avec un échec
ARCANE_ADD_TEST(unittest_fail testUnitTestFail.arc)
set_tests_properties(unittest_fail PROPERTIES WILL_FAIL TRUE)
if (TARGET arcane_mpi)
  set_tests_properties(unittest_fail_4proc PROPERTIES WILL_FAIL TRUE)
endif()

#############
# IOS TESTS #
#############
set(_VTK_FOUND_BOOL false)
if(vtkIOXML_FOUND)
  set(_VTK_FOUND_BOOL true)
endif()
set(_XDMF_FOUND_BOOL false)
if(vtkIOXdmf2_FOUND)
  set(_XDMF_FOUND_BOOL true)
endif()
configure_file(${ARCANE_ARC_PATH}/testIos.cmake ${ARCANE_TEST_PATH}/testIos.arc @ONLY)
ARCANE_ADD_TEST_SEQUENTIAL(ios ${ARCANE_TEST_PATH}/testIos.arc)
# Désactive temporairement ce test tant qu'on ne détecte pas correctement la dimension
# du maillage
#arcane_add_test_sequential(ios_msh testIos-msh.arc)
arcane_add_test_sequential(ios_msh2 testIos-msh2.arc)
arcane_add_test_sequential(ios_msh3 testIos-msh3.arc)
arcane_add_test_sequential(ios_msh4 testIos-msh4.arc)
arcane_add_test_sequential(ios_msh5 testIos-msh5.arc)
if (ARCANE_DEFAULT_PARTITIONER_IS_METIS)
  arcane_add_test_parallel_thread(ios_msh4 testIos-msh4.arc 4)
  arcane_add_test_parallel_thread(ios_msh5 testIos-msh5.arc 5)
endif()
if(vtkIOXML_FOUND)
  ARCANE_ADD_TEST_SEQUENTIAL(ios_vtu testIos-vtu.arc)
endif()
if(vtkIOXdmf2_FOUND)
   ARCANE_ADD_TEST_SEQUENTIAL(ios_xmf testIos-xmf.arc)
endif()

###############
# HYODA TESTS #
###############
ARCANE_ADD_TEST_PARALLEL(hyoda testHyoda.arc 1)

ARCANE_ADD_TEST(checkpoint_basic testCheckpoint-3.arc -c 3 -m 5)
ARCANE_ADD_TEST(checkpoint_basic2-v1 testCheckpoint-basic2-v1.arc -c 3 -m 5)
ARCANE_ADD_TEST(checkpoint_basic2 testCheckpoint-basic2.arc -c 3 -m 5)
ARCANE_ADD_TEST(checkpoint_basic2-v3 testCheckpoint-basic2-v3.arc -c 3 -m 5)
arcane_add_test(checkpoint_basic_hash testCheckpoint-basic2-v3.arc -c 3 -m 5 -We,ARCANE_HASHDATABASE_DIRECTORY,${CMAKE_CURRENT_BINARY_DIR}/hashdb)
if (LZ4_FOUND)
  arcane_add_test_sequential(checkpoint_basic2-v3-lz4 testCheckpoint-basic2-v3-lz4.arc -c 3 -m 5 -We,ARCANE_OUTPUT_LEVEL,5)
  arcane_add_test_sequential(checkpoint_basic_hash_lz4 testCheckpoint-basic2-v3-lz4.arc -c 3 -m 5 -We,ARCANE_HASHALGORITHM,SHA3_512 -We,ARCANE_HASHDATABASE_DIRECTORY,${CMAKE_CURRENT_BINARY_DIR}/hashdb2)
endif()
if (BZIP2_FOUND)
  arcane_add_test_sequential(checkpoint_basic2-v3-bzip2 testCheckpoint-basic2-v3-bzip2.arc -c 3 -m 5 -We,ARCANE_OUTPUT_LEVEL,5)
endif()
arcane_add_test(checkpoint_basic_ghost5 testCheckpoint-6.arc -c 3 -m 5)

arcane_add_test_parallel_all(amr2 testAMR-2.arc 3 4)
arcane_add_test_parallel_all(amr2_checkpoint testAMR-2-checkpoint.arc 3 4 -c 2)
ARCANE_ADD_TEST(hydro3 testHydro-3.arc -m 50)
arcane_add_test_sequential(hydro3_cartesian testHydro-3-cartesian.arc -m 50)
arcane_add_test_sequential(hydro3_cartesian_small testHydro-3-cartesian-small.arc -m 10 -We,ARCANE_USE_OLD_MESHGENERATOR,1)
arcane_add_test_parallel_thread(hydro3_cartesian_small testHydro-3-cartesian-small.arc 4 -m 10 -We,ARCANE_USE_OLD_MESHGENERATOR,1)
arcane_add_test_script(continue_hydro3_cartesian_small continue_hydro3_cartesian_small.xml)
arcane_add_test_script(continue_hydro3_cartesian_small_4shm continue_hydro3_cartesian_small_4shm.xml)
arcane_add_test_parallel(hydro3_cartesian_8proc testHydro-3-cartesian.arc 8 "-m 50")
arcane_add_test_parallel_all(hydro3_meshservice testHydro-3-meshservice.arc 3 4 -m 50)
arcane_add_test_parallel_all(hydro3_checkpoint_meshservice testHydro-3-checkpoint-meshservice.arc 3 4 -c 3 -m 10)
arcane_add_test(hydro5 testHydro-5.arc -m 50 -We,ARCANE_MASTER_HAS_OUTPUT_FILE,1)
arcane_add_test(hydro5_message_passing_prof testHydro-5.arc -m 50 -We,ARCANE_MESSAGE_PASSING_PROFILING,JSON)
arcane_add_test(hydrosimd5 testHydroSimd-5.arc -m 50)
if(NOT ARCANE_DISABLE_PERFCOUNTER_TESTS)
  if (ARCANE_HAS_LINUX_PERF_COUNTERS)
    arcane_add_test(hydro5_perf_trace testHydro-5.arc -m 50 -We,ARCANE_TRACE_ENUMERATOR,1)
  endif()
  if (Papi_FOUND)
    arcane_add_test(hydro5_papi_trace testHydro-5.arc -m 50 -We,ARCANE_TRACE_ENUMERATOR,1 -We,ARCANE_PERFORMANCE_COUNTER_SERVICE,Papi)
  endif()
  arcane_add_test(hydrosimd5_trace testHydroSimd-5.arc -m 50 -We,ARCANE_TRACE_ENUMERATOR,1)
endif()

arcane_add_accelerator_test_sequential(hydro_accelerator5 testHydroAccelerator-5.arc -m 50)
arcane_add_accelerator_test_parallel(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
arcane_add_accelerator_test_parallel_thread(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
arcane_add_accelerator_test_message_passing_hybrid(hydro_accelerator5 CASE_FILE testHydroAccelerator-5.arc NB_SHM 2 NB_MPI 2 ARGS -m 50)
arcane_add_test(hydro_accelerator5 testHydroAccelerator-5.arc -m 50)
arcane_add_test_sequential_task(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
arcane_add_test_parallel_thread(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
arcane_add_test_message_passing_hybrid(hydro_accelerator5 CASE_FILE testHydroAccelerator-5.arc NB_SHM 2 NB_MPI 2 ARGS -m 50)
if (ARCANE_ACCELERATOR_RUNTIME_NAME STREQUAL cuda)
  arcane_add_accelerator_test_sequential(hydro_accelerator5_cupti testHydroAccelerator-5.arc -m 20 -We,ARCANE_CUPTI_LEVEL,1)
endif()

arcane_add_accelerator_test_sequential(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc -m 50)
arcane_add_test_sequential(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc -m 50)
arcane_add_test_sequential_task(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc 4 -m 50)

arcane_add_test_sequential_task(hydrosimd5 testHydroSimd-5.arc 4 -m 50)
ARCANE_ADD_TEST_PARALLEL(hydro5-3proc_rep4 testHydro-5.arc 12 -m 50 -R 4)
ARCANE_ADD_TEST_PARALLEL(hydro5-1proc_rep4 testHydro-5.arc 4 -m 50 -R 4)
ARCANE_ADD_TEST(hydro5-listing testHydro-listing.arc -m 50)
ARCANE_ADD_TEST_PARALLEL(hydro5_4proc_3sd testHydro-5.arc 4 -m 50 -A,P=3 -arcane_opt idle_service ParallelTestIdleService)
ARCANE_ADD_TEST(hydro_backward testHydro-back.arc -m 25)
ARCANE_ADD_TEST_PARALLEL_THREAD(hydro5 testHydro-5.arc 4 -m 50)
ARCANE_ADD_TEST_PARALLEL_MPITHREAD(hydro5 testHydro-5.arc 3 4 -m 50)

if(NOT ARCANE_DISABLE_PERFCOUNTER_TESTS)
  if(Papi_FOUND)
    arcane_add_test_sequential(hydro5_papi testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Papi)
    arcane_add_test_sequential_task(hydro5_papi testHydro-5.arc 4 -m 50 -We,ARCANE_PROFILING,Papi)
  endif()
endif()

if (LibUnwind_FOUND)
  arcane_add_test_sequential(hydro5_prof testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Prof)
  arcane_add_test_sequential_task(hydro5_prof testHydro-5.arc 4 -m 50 -We,ARCANE_PROFILING,Prof)
  arcane_add_test_sequential(hydro5_papi_backtrace testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Prof -We,ARCANE_PROFILING_STACKUNWINDING,backtrace)
endif()
if(ARCANE_USE_MPC)
  ARCANE_ADD_TEST_PARALLEL(hydro5_mpc_pthread testHydro-5.arc 4 -P -m=pthread -m 50)
  ARCANE_ADD_TEST_PARALLEL(hydro5_mpc_2p testHydro-5.arc 8 -p=2 -m 50)
endif(ARCANE_USE_MPC)
arcane_add_test(hydro1 testHydro-1.arc "-m 25")
arcane_add_test_sequential(hydro1_ucd testHydro-1-UCD.arc "-m 25")
if (HDF5_FOUND)
  arcane_add_test_parallel_all(hydro1_vtkhdf testHydro-1-vtkhdf.arc 4 3 "-m 50")
  arcane_add_test_parallel_all(hydro1_vtkhdfv2 testHydro-1-vtkhdfv2.arc 4 3 "-m 50")
  arcane_add_test_parallel_all(hydro1_vtkhdfv2_backward testHydro-1-vtkhdfv2-backward.arc 4 3 "-m 58")
endif()
ARCANE_ADD_TEST(hydro_depend1 testHydroDepend-1.arc "-m 25")
ARCANE_ADD_TEST(hydro2 testHydro-2.arc "-m 25")

if (HDF5_FOUND)
  ARCANE_ADD_TEST(hydro1_checkpoint testHydro-1-checkpoint.arc -c 2 -m 25)
  ARCANE_ADD_TEST_SEQUENTIAL(hydro6_checkpoint testHydro-6.arc -c 2 -m 25)
  ARCANE_ADD_TEST_PARALLEL(hydro1_checkpoint_3proc_rep4 testHydro-1-checkpoint.arc 12 -c 2 -m 25 -R 4)
endif()

arcane_add_test_sequential(numarray1 testNumArray-1.arc)
arcane_add_test_sequential_task(numarray1 testNumArray-1.arc 4)
arcane_add_accelerator_test_sequential(numarray1 testNumArray-1.arc)

arcane_add_test_sequential(runqueue1 testRunQueue-1.arc)
arcane_add_test_sequential_task(runqueue1 testRunQueue-1.arc 4)
arcane_add_accelerator_test_sequential(runqueue1 testRunQueue-1.arc)

arcane_add_test_sequential(acceleratorviews1 testAcceleratorViews-1.arc)
arcane_add_test_sequential_task(acceleratorviews1 testAcceleratorViews-1.arc 4)
arcane_add_accelerator_test_sequential(acceleratorviews1 testAcceleratorViews-1.arc)

arcane_add_test_sequential(accelerator_reduce1 testAcceleratorReduce-1.arc)
arcane_add_test_sequential_task(accelerator_reduce1 testAcceleratorReduce-1.arc 4)
arcane_add_accelerator_test_sequential(accelelerator_reduce1 testAcceleratorReduce-1.arc)
arcane_add_accelerator_test_sequential(accelelerator_reduce1_atomic testAcceleratorReduce-1-atomic.arc)

arcane_add_test_sequential(accelerator_material1 testAcceleratorMaterials-1.arc)
arcane_add_test_sequential_task(accelerator_material1 testAcceleratorMaterials-1.arc 4)
arcane_add_accelerator_test_sequential(accelerator_material1 testAcceleratorMaterials-1.arc)

arcane_add_test_sequential(memorycopy1 testMemoryCopy-1.arc)
arcane_add_test_sequential_task(memorycopy1 testMemoryCopy-1.arc 4)
arcane_add_accelerator_test_sequential(memorycopy1 testMemoryCopy-1.arc)

arcane_add_test_sequential(multi_memorycopy1 testMultiMemoryCopy-1.arc)
arcane_add_test_sequential_task(multi_memorycopy1 testMultiMemoryCopy-1.arc 4)
arcane_add_accelerator_test_sequential(multi_memorycopy1 testMultiMemoryCopy-1.arc)

arcane_add_test_sequential(miniweather_orig1 testMiniWeatherOriginal.arc)
arcane_add_test_sequential(miniweather_array_seq1 testMiniWeatherArraySequential.arc)
arcane_add_test_sequential(miniweather_array1_right_layout testMiniWeatherArray.arc)
arcane_add_test_sequential(miniweather_array1_left_layout testMiniWeatherArrayLeftLayout.arc)
arcane_add_test_sequential_task(miniweather_array1 testMiniWeatherArray.arc 4)
arcane_add_accelerator_test_sequential(miniweather_array1_right_layout testMiniWeatherArray.arc)
arcane_add_accelerator_test_sequential(miniweather_array1_left_layout testMiniWeatherArrayLeftLayout.arc)
arcane_add_accelerator_test_sequential(miniweather_array1_device testMiniWeatherArrayDevice.arc)
arcane_add_test_sequential(pdes_random_number_generator_service testPDESRandomNumberGenerator.arc)
arcane_add_test_sequential(simple_csv_output_service testSimpleCsvOutputService.arc)
arcane_add_test_sequential(simple_csv_comparator_service testSimpleCsvComparatorService.arc)


if(ARCANE_HAS_GEOMETRY)
  ARCANE_ADD_TEST(geometry testGeom.arc)
endif()
if(ARCANE_HAS_SOLVERS)
  ARCANE_ADD_TEST_PARALLEL(solver testSolver.arc 1)
  ARCANE_ADD_TEST_PARALLEL(solver testSolver.arc 4)
endif()

if (ARCANE_HAS_DOTNET_TESTS)
  if (MONO_EXEC)
    # TODO: faire aussi ces tests dotnet_core
    arcane_add_test_sequential(direct_exec1_cs none.arc -We,ARCANE_SIMPLE_EXECUTOR,1 -Z)
    arcane_add_test_sequential(direct_exec2_cs none.arc --dotnet-assembly=${ARCANE_TEST_DOTNET_ASSEMBLY} -We,ARCANE_USE_DOTNET_WRAPPER,1 -A,DotNetExecuteClassName=TestClass -A,DotNetExecuteMethodName=Test1 -Z)
    arcane_add_test_sequential(direct_exec3_cs none.arc -A,DotNetExecuteClassName=TestClass -A,DotNetExecuteMethodName=Test1 -Z)
  endif()
  arcane_add_csharp_test_sequential(mesh_modification1_cs testMeshModification-1.arc -m 15)
  if (MONO_EXEC)
    arcane_add_test_sequential(mesh_modification2_cs testMeshModification-1.arc -A,DotNetExecuteClassName=TestClass -A,DotNetExecuteMethodName=Test2 -Z)
  endif()
  arcane_add_csharp_test_parallel(hydro_cs testSimpleHydroCS-1.arc 4 -m 15 -A,MessagePassingService=Mpi)
  arcane_add_csharp_test_sequential(hydro_cs testSimpleHydroCS-1.arc -m 15)
  if (ARCANE_HAS_TASKS)
    arcane_add_csharp_test_sequential(hydro_cs_task testSimpleHydroCS-1.arc -m 15 -K 4)
  endif()
endif()

arcane_add_test_parallel(loadbalance_test1 testLoadBalanceHydro-MeshPartitionerTester.arc 4 -m 30)
arcane_add_test_parallel(loadbalance_test1_d2 testLoadBalanceHydro-MeshPartitionerTester2.arc 4 -m 30)
arcane_add_test_parallel(loadbalance_test1 testLoadBalanceHydro-MeshPartitionerTester.arc 12 -m 30)
arcane_add_test_parallel(loadbalance_test1_3exchange_v1 testLoadBalanceHydro-MeshPartitionerTester.arc 6 -We,ARCANE_NB_EXCHANGE=3 -m 30)
arcane_add_test_parallel(loadbalance_test1_3exchange_v2 testLoadBalanceHydro-MeshPartitionerTester.arc 6 -We,ARCANE_NB_EXCHANGE,3 -We,ARCANE_MESH_EXCHANGE_VERSION,2 -m 30)
if(Parmetis_FOUND)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_metis1 testLoadBalanceHydro-Metis.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL_THREAD(loadbalance_metis3 testLoadBalanceHydro-Metis3.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_metis1_no_node testLoadBalanceHydro-Metis2.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_metis1_rep3 testLoadBalanceHydro-Metis.arc 12 -m 30 -R 3)
endif()
if(Zoltan_FOUND)
  ARCANE_ADD_TEST_PARALLEL(zoltan1 testPartition-zoltan.arc 4)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_zoltanH testLoadBalanceHydro-Zoltan-hypergraphe.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_zoltanG testLoadBalanceHydro-Zoltan-geometric.arc 4 -m 30)
endif()
if(PTScotch_FOUND)
  ARCANE_ADD_TEST_PARALLEL(ptscotch1 testPartition-ptscotch.arc 4)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_ptscotch1 testLoadBalanceHydro-PTScotch.arc 4 -m 80)
endif()

ARCANE_ADD_TEST_PARALLEL_THREAD(thread1 testThread-1.arc 1 -m 15)
arcane_add_test_sequential(task1_0 testTask-1.arc 1 -m 5)
arcane_add_test_sequential_task(task1 testTask-1.arc 1 -m 5)
arcane_add_test_sequential_task(task1 testTask-1.arc 4 -m 5)
arcane_add_test_sequential_task(task1_setoptions testTask-1.arc 4 -m 5 -A,ParallelLoopGrainSize=4 -A,ParallelLoopPartitioner=static)
arcane_add_test_sequential_task(task1_loop_profile testTask-1.arc 4 -m 5 -We,ARCANE_LOOP_PROFILING_LEVEL,2)
if(HWLoc_FOUND)
  arcane_add_test_sequential_task(task1_bind testTask-1.arc 4 -m 5 -A,ThreadBindingStrategy=Simple)
endif()
arcane_add_test_sequential_task(task1 testTask-1.arc 8 -m 5)
arcane_add_test_sequential_task(task1 testTask-1.arc 0 -m 5)
if (ARCANE_HAS_DOTNET_TESTS)
  if (ARCANE_HAS_TASKS)
    arcane_add_csharp_test_sequential(task2_cs testTask-2.arc -m 5 -K 4)
  endif()
endif()
arcane_add_test_sequential_task(hydro5 testHydro-5.arc 0 -m 50)
arcane_add_test_sequential_task(hydro5 testHydro-5.arc 4 -m 50)

if(GEOMETRYKERNEL_FOUND)
  ARCANE_ADD_TEST_PARALLEL(corefinement testParallelCorefinement.arc 1)
  ARCANE_ADD_TEST_PARALLEL(corefinement testParallelCorefinement.arc 4)
endif()

if(VTK_FOUND AND MPI_FOUND)
  ADD_TEST(case_part_vtu ${ARCANE_TEST_PATH}/test-CasePartVtu.sh)
endif()

add_test(convert_mesh1 ${ARCANE_CONVERT_MESH_COMMAND} -f convert_out_mesh.vtk --writer VtkLegacyMeshWriter ${ARCANE_TEST_PATH}/plan.vtk)
arcane_add_test(tied_interface1 testTiedInterface-1.arc "-m 10")
arcane_add_test(tied_interface2 testTiedInterface-2.arc "-m 10")
arcane_add_test(tied_interface1_2D testTiedInterface2D-1.arc "-m 10")

######################################################
# ALEPH STD TEST                                     #
######################################################

if(TARGET arcane_aleph_hypre)
  ARCANE_ADD_TEST_PARALLEL(aleph testAleph.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph testAleph.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph testAleph.arc 4)
  ARCANE_ADD_TEST_PARALLEL(aleph_hypre testAlephHypre.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_hypre testAlephHypre.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_hypre testAlephHypre.arc 4)
endif()

if(TARGET arcane_aleph_trilinos)
  ARCANE_ADD_TEST_PARALLEL(aleph_trilinos testAlephTrilinos.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_trilinos testAlephTrilinos.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_trilinos testAlephTrilinos.arc 4)
endif()

if(TARGET arcane_aleph_petsc)
  ARCANE_ADD_TEST_PARALLEL(aleph_petsc testAlephPETSc.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_petsc testAlephPETSc.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_petsc testAlephPETSc.arc 4)
endif()

######################################################
# ALEPH ODD TEST                                     #
######################################################

if(TARGET arcane_aleph_hypre)
  ARCANE_ADD_TEST_PARALLEL(aleph_hypre_odd testAlephHypreOdd.arc 3)
  ARCANE_ADD_TEST_PARALLEL(aleph_hypre_odd testAlephHypreOdd.arc 4)
  ARCANE_ADD_TEST_PARALLEL(aleph_hypre_odd testAlephHypreOdd.arc 7)
  ARCANE_ADD_TEST_PARALLEL(aleph_hypre_odd testAlephHypreOdd.arc 8)
endif()

if(TARGET arcane_aleph_trilinos)
  ARCANE_ADD_TEST_PARALLEL(aleph_trilinos_odd testAlephTrilinosOdd.arc 3)
  ARCANE_ADD_TEST_PARALLEL(aleph_trilinos_odd testAlephTrilinosOdd.arc 4)
  ARCANE_ADD_TEST_PARALLEL(aleph_trilinos_odd testAlephTrilinosOdd.arc 7)
  ARCANE_ADD_TEST_PARALLEL(aleph_trilinos_odd testAlephTrilinosOdd.arc 8)
endif()

if(TARGET arcane_aleph_petsc)
  ARCANE_ADD_TEST_PARALLEL(aleph_petsc_odd testAlephPETScOdd.arc 3)
  ARCANE_ADD_TEST_PARALLEL(aleph_petsc_odd testAlephPETScOdd.arc 4)
  ARCANE_ADD_TEST_PARALLEL(aleph_petsc_odd testAlephPETScOdd.arc 7)
  ARCANE_ADD_TEST_PARALLEL(aleph_petsc_odd testAlephPETScOdd.arc 8)
endif()


#################################################################
# ALEPH DELETE TEST                                             #
#################################################################

if(TARGET arcane_aleph_hypre)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_hypre testAlephHypreDelete.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_hypre testAlephHypreDelete.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_hypre testAlephHypreDelete.arc 4)
endif()

if(TARGET arcane_aleph_trilinos)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_trilinos testAlephTrilinosDelete.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_trilinos testAlephTrilinosDelete.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_trilinos testAlephTrilinosDelete.arc 4)
endif()

if(TARGET arcane_aleph_petsc)
  ARCANE_ADD_TEST_PARALLEL(aleph_delete_petsc testAlephPETScDelete.arc 1)
endif()

#################################################################
# ALEPH KAPPA TEST                                              #
#################################################################

if(TARGET arcane_aleph_hypre)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa testAleph.arc 2 -arcane_opt nb_sub_domain 1 -arcane_opt idle_service AlephKappa)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_hypre testAlephHypre.arc 2 -arcane_opt nb_sub_domain 1 -arcane_opt idle_service AlephKappa)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa testAleph.arc 4 -arcane_opt nb_sub_domain 2 -arcane_opt idle_service AlephKappa)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_hypre testAlephHypre.arc 4 -arcane_opt nb_sub_domain 2 -arcane_opt idle_service AlephKappa)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_hypre_odd testAlephHypreOdd.arc 4 -arcane_opt nb_sub_domain 2 -arcane_opt idle_service AlephKappa)
endif()

if(TARGET arcane_aleph_trilinos)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_trilinos testAlephTrilinos.arc 2 -arcane_opt nb_sub_domain 1 -arcane_opt idle_service AlephKappa)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_trilinos testAlephTrilinos.arc 4 -arcane_opt nb_sub_domain 2 -arcane_opt idle_service AlephKappa)
endif()

if(TARGET arcane_aleph_petsc)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_petsc testAlephPETSc.arc 2 -arcane_opt nb_sub_domain 1 -arcane_opt idle_service AlephKappa)
  ARCANE_ADD_TEST_PARALLEL(aleph_kappa_petsc testAlephPETSc.arc 4 -arcane_opt nb_sub_domain 2 -arcane_opt idle_service AlephKappa)
endif()

####################
# ALEPH+INDEX TEST #
####################
if(TARGET arcane_aleph_hypre)
  ARCANE_ADD_TEST_PARALLEL(aleph_index testAlephIndex.arc 1)
  ARCANE_ADD_TEST_PARALLEL(aleph_index_parallel testAlephIndex.arc 2)
  ARCANE_ADD_TEST_PARALLEL(aleph_index_parallel testAlephIndex.arc 4)
endif()

####################
# MULTI ALEPH TEST #
####################
ARCANE_ADD_TEST_PARALLEL(aleph_multi testAlephMulti.arc 1)
ARCANE_ADD_TEST_PARALLEL(aleph_multi testAlephMulti.arc 2)
ARCANE_ADD_TEST_PARALLEL(aleph_multi testAlephMulti.arc 4)
ARCANE_ADD_TEST_PARALLEL(aleph_multi testAlephMulti.arc 6)
ARCANE_ADD_TEST_PARALLEL(aleph_multi testAlephMulti.arc 8)

#################################################################
#################################################################

# Test les comparaisons bit à bit entre deux exécutions
arcane_add_test_script(compare_seq_seq compare_seq_seq.xml)
arcane_add_test_script(compare_seq_seq_v3 compare_seq_seq_v3.xml)
if (LZ4_FOUND)
  arcane_add_test_script(compare_seq_seq_lz4 compare_seq_seq_lz4.xml)
  arcane_add_test_script(compare_seq_seq_lz4_v3 compare_seq_seq_lz4_v3.xml)
endif()
if (BZIP2_FOUND)
  arcane_add_test_script(compare_seq_seq_bzip2 compare_seq_seq_bzip2.xml)
  arcane_add_test_script(compare_seq_seq_bzip2_v3 compare_seq_seq_bzip2_v3.xml)
endif()
if (TARGET arcane_mpi)
  arcane_add_test_script(compare_seq_par compare_seq_par.xml)
  arcane_add_test_script(compare_par_par compare_par_par.xml)
  arcane_add_test_script(compare_seq_par_v3 compare_seq_par_v3.xml)
  arcane_add_test_script(compare_par_par_v3 compare_par_par_v3.xml)
endif()
arcane_add_test_script(checkpoint_verifier_seq_v3 checkpoint_verifier_seq_v3.xml)
arcane_add_test_script(checkpoint_verifier_seq_4pe_v3 checkpoint_verifier_seq_4pe_v3.xml)
arcane_add_test_script(checkpoint_verifier_4pe_v3 checkpoint_verifier_4pe_v3.xml)

#################################################################
#################################################################

# Test les reprises en changeant le nombre de sous-domaines
if (ARCANE_DEFAULT_PARTITIONER_IS_METIS)
  arcane_add_test_script(checkpoint_change_n8_n5_thread checkpoint_change_n8_n5_thread.xml)
endif()
if (TARGET arcane_mpi)
  arcane_add_test_script(checkpoint_change_n4_n2 checkpoint_change_n4_n2.xml)
  arcane_add_test_script(checkpoint_change_n8_n5 checkpoint_change_n8_n5.xml)
  arcane_add_test_script(checkpoint_change_n7_n4 checkpoint_change_n7_n4.xml)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
