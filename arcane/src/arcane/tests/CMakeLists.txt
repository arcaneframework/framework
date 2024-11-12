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

list(APPEND TEST_DIRS geometry)
list(APPEND TEST_LIBS arcane_geometry)

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
  accelerator/AtomicUnitTest.cc
  accelerator/SimpleHydroAcceleratorService.cc
  accelerator/MiniWeatherArray.cc
  accelerator/MemoryCopyUnitTest.cc
  accelerator/MultiMemoryCopyUnitTest.cc
  accelerator/NumArrayUnitTest.cc
  accelerator/AcceleratorItemInfoUnitTest.cc
  accelerator/AcceleratorReduceUnitTest.cc
  accelerator/AcceleratorScanUnitTest.cc
  accelerator/AcceleratorFilterUnitTest.cc
  accelerator/AcceleratorPartitionerUnitTest.cc
  accelerator/AcceleratorSorterUnitTest.cc
  accelerator/RunQueueUnitTest.cc
  accelerator/AcceleratorMathUnitTest.cc
  accelerator/AcceleratorViewsUnitTest.cc
  accelerator/ArcaneTestStandaloneAcceleratorMng.cc
  accelerator/MeshMaterialAcceleratorUnitTest.cc
)
if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND ARCANE_SOURCES
    MDVariableUnitTest.cc
    accelerator/MiniWeatherOriginalSequential.cc
    accelerator/MiniWeatherArraySequential.cc
    )
  list(APPEND ARCANE_SOURCES ${ARCANE_ACCELERATOR_SOURCES})
  arcane_accelerator_add_source_files(${ARCANE_ACCELERATOR_SOURCES})
endif()

if (TARGET hiredis::hiredis)
  list(APPEND ARCANE_SOURCES RedisUnitTest.cc)
endif()

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
  target_compile_definitions(arcane_tests_exec PRIVATE ARCANE_HAS_POLYHEDRAL_MESH_TOOLS)
  target_compile_definitions(arcane_tests_lib PUBLIC ARCANE_HAS_POLYHEDRAL_MESH_TOOLS)
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
  if (ARCANE_ENABLE_DOTNET_PYTHON_WRAPPER)
    # Cette variable est utilisée lors de la génération du fichier '.csproj'
    set(_HAS_PYTHON_WRAPPER "true")
  endif()
  configure_file(ArcaneTest.csproj.in ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneTest/ArcaneTest.csproj @ONLY)

  add_custom_command(OUTPUT ${CSOUTPATH}/SimpleHydroCS_axl.cs
    DEPENDS ${CSPATH}/SimpleHydroCS.axl ${ARCANE_AXLSTAR_DEPEND}
    COMMAND ${ARCANE_AXL2CC}
    ARGS -i arcane/tests/. --lang c\# -o ${CSOUTPATH} ${CSPATH}/SimpleHydroCS.axl)

  add_custom_command(OUTPUT ${CSOUTPATH}/PerfectGasEOS_axl.cs
    DEPENDS ${CSPATH}/PerfectGasEOS.axl ${ARCANE_AXLSTAR_DEPEND}
    COMMAND ${ARCANE_AXL2CC}
    ARGS -i arcane/tests/. --lang c\# -o ${CSOUTPATH} ${CSPATH}/PerfectGasEOS.axl)

  add_custom_command(OUTPUT ${CSOUTPATH}/MeshModification_axl.cs
    DEPENDS ${CSPATH}/MeshModification.axl ${ARCANE_AXLSTAR_DEPEND}
    COMMAND ${ARCANE_AXL2CC}
    ARGS -i arcane/tests/. --lang c\# -o ${CSOUTPATH} ${CSPATH}/MeshModification.axl)

  add_custom_target(arcane_test_cs_generate ALL DEPENDS
    "${CSOUTPATH}/SimpleHydroCS_axl.cs"
    "${CSOUTPATH}/PerfectGasEOS_axl.cs"
    "${CSOUTPATH}/MeshModification_axl.cs"
  )

  arcane_add_global_csharp_target(arcane_test_cs
    BUILD_DIR ${LIBRARY_OUTPUT_PATH}
    ASSEMBLY_NAME ArcaneTest.dll
    PROJECT_PATH ${ARCANE_CSHARP_PROJECT_PATH}/ArcaneTest
    PROJECT_NAME ArcaneTest.csproj
    MSBUILD_ARGS ${ARCANE_MSBUILD_ARGS}
    DEPENDS
    "${CSPATH}/MeshModification.cs"
    "${CSPATH}/DotNetTimeHistoryCurveWriter.cs"
    "${CSPATH}/DotNetCaseFunctionProvider.cs"
    "${CSPATH}/DotNetDataWriter.cs"
    "${CSPATH}/DotNetTestSingleton.cs"
    "${CSPATH}/SimpleHydro.cs"
    "${CSPATH}/PerfectGasEOS.cs"
    "${CSPATH}/Launcher.cs"
    arcane_test_cs_generate
    DOTNET_TARGET_DEPENDS dotnet_wrapper_services dotnet_wrapper_launcher
    )

  arcane_add_global_csharp_target(arcane_test_cs_exe
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

if (ARCANE_HAS_ACCELERATOR_API)
  target_link_libraries(arcane_tests_lib PRIVATE arcane_accelerator)
endif()
target_link_libraries(arcane_tests_lib PRIVATE arcane_accelerator_core)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

set(ARCANE_TEST_WORKDIR ${CMAKE_BINARY_DIR}/src/arcane/tests)
set(ARCANE_TEST_CASEPATH ${TEST_PATH})
set(ARCANE_TEST_EXECNAME arcane_tests_exec)
set(ARCANE_TEST_DOTNET_ASSEMBLY ${LIBRARY_OUTPUT_PATH}/ArcaneTest.dll)
include(ArcaneTests.cmake)

# ----------------------------------------------------------------------------

function(ARCANE_COPY_MESH orig_mesh_name mesh_name ext)
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ARCANESRCROOT}/maillages/${orig_mesh_name}.${ext}
    ${ARCANE_TEST_PATH}/${mesh_name}.${ext})
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ARCANESRCROOT}/maillages/${orig_mesh_name}.vtkfaces.${ext}
    ${ARCANE_TEST_PATH}/${mesh_name}.${ext}faces.${ext})
endfunction()
function(ARCANE_COPY_MESH_DIRECT orig_mesh_name)
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ARCANESRCROOT}/maillages/${orig_mesh_name}
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
arcane_copy_mesh_direct(tube2x2x4.vtk)
arcane_copy_mesh_direct(tube2x2x4.vtkfaces.vtk)
arcane_copy_mesh_direct(faultx1_2x1x1.vtk)
arcane_copy_mesh_direct(faultx1_2x1x1.vtkfaces.vtk)
arcane_copy_mesh_direct(cross_a_2x1x1.vtk)
arcane_copy_mesh_direct(cross_a_2x1x1.vtkfaces.vtk)
arcane_copy_mesh_direct(fan_2x1x3.vtk)
arcane_copy_mesh_direct(fan_2x1x3.vtkfaces.vtk)
arcane_copy_mesh_direct(cross_a_2x1x1.vtu)
arcane_copy_mesh_direct(cross_a_2x1x1.vtufaces.vtu)
arcane_copy_mesh_direct(plancher.msh)
arcane_copy_mesh_direct(hex_tetra_pyramics.msh)
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
ARCANE_ADD_TEST_SEQUENTIAL(simd1 testSimd-1.arc -We,ARCANE_CHECK_SIMDPADDING,2)
ARCANE_ADD_TEST_SEQUENTIAL(singleton1 testSingletonService-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(array1 testArcane-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(configuration1 testConfiguration-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(properties1 testProperties-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(itemvector1 testItemVector-1.arc)
ARCANE_ADD_TEST_SEQUENTIAL(variable testVariable-1.arc)
if (LZ4_FOUND)
  ARCANE_ADD_TEST_SEQUENTIAL(variable_lz4 testVariable-1-lz4.arc)
endif()
if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test_sequential(mdvariable testMDVariable-1.arc)
endif()
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
# Mettre à vrai pour tester Redis mais il faut que le server tourne
# (on peut tester cela avec la commande 'redis-cli PING')
set(ARCANE_ENABLE_REDIS_TEST FALSE)
if (hiredis_FOUND AND ARCANE_ENABLE_REDIS_TEST)
  # Pour ce test on a aussi besoin qu'une base redis soit accessible
  # Comme il n'y a pas de moyen simple de le savoir, ce test n'est pas actif
  # par défaut.
  arcane_add_test_sequential(redis1 testRedis-1.arc)
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
arcane_add_test(direct_exec_cartesian testRunDirect-1.arc -D TestRunDirectCartesian)

add_test(dump_all_internal ${ARCANE_TEST_DRIVER} launch -We,STDENV_CODE_NAME,Arcane -arcane_opt arcane_all_internal)
add_test(dump_internal ${ARCANE_TEST_DRIVER} launch -We,STDENV_CODE_NAME,Arcane -A,MessagePassingService=Sequential -arcane_opt arcane_internal)
add_test(dump_database ${ARCANE_TEST_DRIVER} launch -We,STDENV_CODE_NAME,Arcane -arcane_opt arcane_database)

# ----------------------------------------------------------------------------

if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test_sequential(standalone_accelerator_testsum dummy.arc "-A,StandaloneAcceleratorMethod=TestSum")
  arcane_add_test_sequential(standalone_accelerator_testbinop dummy.arc "-A,StandaloneAcceleratorMethod=TestBinOp")
  arcane_add_accelerator_test_sequential(standalone_accelerator_testsum dummy.arc "-A,StandaloneAcceleratorMethod=TestSum")
  arcane_add_accelerator_test_sequential(standalone_accelerator_testbinop dummy.arc "-A,StandaloneAcceleratorMethod=TestBinOp")
  arcane_add_accelerator_test_sequential(standalone_accelerator_testemptykernel dummy.arc "-A,StandaloneAcceleratorMethod=TestEmptyKernel")
endif()

arcane_add_test_sequential(standalone_subdomain1 dummy.arc "-A,StandaloneSubDomainMethod=Test1")
arcane_add_test_sequential(standalone_subdomain2 dummy.arc "-A,StandaloneSubDomainMethod=Test2")
arcane_add_test_parallel(standalone_subdomain2 dummy.arc 4 "-A,StandaloneSubDomainMethod=Test2")
arcane_add_test_parallel(standalone_subdomain2_legacy_partition_init dummy.arc 4 "-A,StandaloneSubDomainMethod=Test2" "-We,ARCANE_USE_LEGACY_INIT_LOADBALANCEMNG,1")

# ----------------------------------------------------------------------------

# Macro pour tester le message_passing
macro(add_test_message_passing test_name)
  set(_OPTS -We,MESSAGE_PASSING_TEST,${test_name} -We,ARCCORE_ALLOW_NULL_RANK_FOR_MPI_ANY_SOURCE,1 -arcane_opt direct_test ParallelMngTest null)
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

add_test(serialize_message_list_v1_4proc
  ${ARCANE_TEST_DRIVER} launch -n 4
  -We,MESSAGE_PASSING_TEST,serialize_message_list
  -We,ARCANE_SYNCHRONIZE_LIST_VERSION,1
  -arcane_opt direct_test ParallelMngTest null)

# ----------------------------------------------------------------------------

ARCANE_ADD_TEST_PARALLEL(parallel testParallel-1.arc 4)
ARCANE_ADD_TEST_PARALLEL(parallel2 testParallel-2.arc 4)
arcane_add_test_parallel(parallel2_synchronize testParallel-synchronize1.arc 4)
arcane_add_test_parallel_thread(parallel2_synchronize testParallel-synchronize1.arc 4)
arcane_add_test_parallel(parallel2_synchronize_compare testParallel-synchronize1.arc 4 -We,ARCANE_AUTO_COMPARE_SYNCHRONIZE,3)
arcane_add_test_parallel(parallel2_synchronize_legacymulti testParallel-synchronize1.arc 4 -We,ARCANE_USE_LEGACY_MULTISYNCHRONIZE,1)
arcane_add_test_parallel(parallel2_synchronize_v1 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,1)
arcane_add_test_parallel(parallel2_synchronize_v2 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,2)
arcane_add_test_parallel(parallel2_synchronize_v3 testParallel-synchronize1.arc 4 -We,ARCANE_SYNCHRONIZE_VERSION,3 -We,ARCANE_CHECK_SYNCHRONIZE_COHERENCE,1)
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

# Le test suivant ne fonctionne en mode hybride que si le
# partitionneur par défaut est ParMetis
if (ARCANE_DEFAULT_PARTITIONER_IS_METIS)
  arcane_add_test_parallel_all(face-submesh_2d testSubMesh2-2D.arc 2 2)
else()
  arcane_add_test(face-submesh_2d testSubMesh2-2D.arc)
  arcane_add_test_parallel(face-submesh_2d testSubMesh2-2D.arc 2 2)
endif()

ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_meshpartitionertester1 testMeshCriteriaLoadBalanceMng-MeshPartitionerTester-1.arc 8 -m 10)
ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_meshpartitionertester_4_1 testMeshCriteriaLoadBalanceMng-MeshPartitionerTester-4-1.arc 4 -m 10)

if(PARTITIONER_FOUND)
  ARCANE_ADD_TEST_PARALLEL(loadbalance testLoadBalance-1.arc 4)
  ARCANE_ADD_TEST_PARALLEL_THREAD(loadbalance testParticle.arc 4)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_rep3 testLoadBalance-1.arc 12 -R 3)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_collective testLoadBalance-1.arc 4 -We,ARCANE_MESH_EXCHANGE_USE_COLLECTIVE,1 -We,ARCANE_PRINT_CPUAFFINITY,1)

  ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_defaultpartitioner1 testMeshCriteriaLoadBalanceMng-DefaultPartitioner-1.arc 8 -m 10)
  ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_defaultpartitioner_4_1 testMeshCriteriaLoadBalanceMng-DefaultPartitioner-4-1.arc 4 -m 10)
  ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_defaultpartitioner_meshpartitionertester1 testMeshCriteriaLoadBalanceMng-DefaultPartitioner-MeshPartitionerTester-1.arc 8 -m 10)
  ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_defaultpartitioner_meshpartitionertester_4_1 testMeshCriteriaLoadBalanceMng-DefaultPartitioner-MeshPartitionerTester-4-1.arc 4 -m 10)

  if (Parmetis_FOUND)
    ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis1 testMeshCriteriaLoadBalanceMng-Metis-1.arc 8 -m 10)
    ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_4_1 testMeshCriteriaLoadBalanceMng-Metis-4-1.arc 4 -m 10)
    if (PTScotch_FOUND)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_ptscotch1 testMeshCriteriaLoadBalanceMng-Metis-PTScotch-1.arc 8 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_ptscotch_4_1 testMeshCriteriaLoadBalanceMng-Metis-PTScotch-4-1.arc 4 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_ptscotch2 testMeshCriteriaLoadBalanceMng-Metis-PTScotch-2.arc 8 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_ptscotch_4_2 testMeshCriteriaLoadBalanceMng-Metis-PTScotch-4-2.arc 4 -m 10)
    endif ()
    if (Zoltan_FOUND)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_zoltan1 testMeshCriteriaLoadBalanceMng-Metis-Zoltan-1.arc 8 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_zoltan_4_1 testMeshCriteriaLoadBalanceMng-Metis-Zoltan-4-1.arc 4 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_zoltan2 testMeshCriteriaLoadBalanceMng-Metis-Zoltan-2.arc 8 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_metis_zoltan_4_2 testMeshCriteriaLoadBalanceMng-Metis-Zoltan-4-2.arc 4 -m 10)
    endif ()
  endif ()

  if (PTScotch_FOUND)
    ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_ptscotch1 testMeshCriteriaLoadBalanceMng-PTScotch-1.arc 8 -m 10)
    ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_ptscotch_4_1 testMeshCriteriaLoadBalanceMng-PTScotch-4-1.arc 4 -m 10)
    if (Zoltan_FOUND)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_ptscotch_zoltan1 testMeshCriteriaLoadBalanceMng-PTScotch-Zoltan-1.arc 8 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_ptscotch_zoltan_4_1 testMeshCriteriaLoadBalanceMng-PTScotch-Zoltan-4-1.arc 4 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_ptscotch_zoltan2 testMeshCriteriaLoadBalanceMng-PTScotch-Zoltan-2.arc 8 -m 10)
      ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_ptscotch_zoltan_4_2 testMeshCriteriaLoadBalanceMng-PTScotch-Zoltan-4-2.arc 4 -m 10)
    endif ()
  endif ()

  if (Zoltan_FOUND)
    ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_zoltan1 testMeshCriteriaLoadBalanceMng-Zoltan-1.arc 8 -m 10)
    ARCANE_ADD_TEST_PARALLEL(meshcriterialoadbalancemng_zoltan_4_1 testMeshCriteriaLoadBalanceMng-Zoltan-4-1.arc 4 -m 10)
  endif ()
endif()

arcane_add_test_parallel_all(particle testParticle.arc 3 4)
arcane_add_test_parallel_all(particle_nonblocking testParticleNonBlocking.arc 3 4)
arcane_add_test_sequential(particle_async testParticleAsync.arc)
arcane_add_test_parallel(particle_async testParticleAsync.arc 4)
ARCANE_ADD_TEST_SEQUENTIAL(voronoi testVoronoi.arc -We,ARCANE_ITEM_TYPE_FILE,voronoi.format)
ARCANE_ADD_TEST_PARALLEL(voronoi testVoronoi.arc 4 -We,ARCANE_ITEM_TYPE_FILE,voronoi.format)
arcane_add_test(mesh testMesh-1.arc -We,ARCANE_DEBUG_VARIABLESYNCHRONIZERCOMPUTELIST,1)
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
ARCANE_ADD_TEST_PARALLEL(graphdof_fourcells testGraph-TwoCells2P.arc 2 -We,ARCANE_GRAPH_CONNECTIVITY_POLICY,1)
ARCANE_ADD_TEST_PARALLEL(graphdof_fourcells testGraph-FourCells.arc 4 -We,ARCANE_GRAPH_CONNECTIVITY_POLICY,1)
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

arcane_add_test(mesh2 testMesh-2.arc "-We,ARCANE_METIS_INPUT_OUTPUT_DIGEST,1")
arcane_add_test_sequential(mesh2_sort_faces testMesh-2-sorted-faces.arc)
arcane_add_test_sequential(mesh2_init_nan testMesh-2.arc "-We,ARCANE_DATA_INIT_POLICY,NAN")
arcane_add_test_sequential(mesh2_init_default testMesh-2.arc "-We,ARCANE_DATA_INIT_POLICY,DEFAULT" "-We,ARCANE_ITEMFAMILY_SHRINK_AFTER_ALLOCATE,1")
arcane_add_test_sequential(mesh2_init_nan_and_default testMesh-2.arc "-We,ARCANE_DATA_INIT_POLICY,NAN_AND_DEFAULT")
arcane_add_test_sequential(mesh2_small testMesh-2-small-vtk.arc)
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
  if (vtkIO_Legacy AND Neo_FOUND)
    arcane_add_test_sequential(polyhedral_vtk1 testMeshPolyhedral-1.arc)
    arcane_add_test_sequential(polyhedral_vtk2 testMeshPolyhedral-2.arc)
    arcane_add_test_sequential(polyhedral_vtk3 testMeshPolyhedral-3.arc)
    arcane_add_test_sequential(polyhedral_vtk4 testMeshPolyhedral-4.arc)
    arcane_add_test_sequential(polyhedral_with_dofs testMeshPolyhedralWithDoFs.arc)
  endif()
endif()
if(MEDFile_FOUND)
  arcane_add_test_sequential(mesh1_med testMesh-med1.arc)
  arcane_add_test(mesh2_med testMesh-med2.arc)
  arcane_add_test(mesh2_med_meshservice testMesh-med2-meshservice.arc)
endif()
ARCANE_ADD_TEST(mesh_quadratic3d_vtk testMesh-quadratic3d-vtk.arc "-We,ARCANE_METIS_INPUT_OUTPUT_DIGEST,1" "-We,ARCANE_METIS_DUMP_GRAPH,1")
ARCANE_ADD_TEST(mesh_tied_interface_2d_1_vtk42 testMesh-tied_interface_2d_1-vtk42.arc)
ARCANE_ADD_TEST(mesh_sphere_vtk42 testMesh-sphere-vtk42.arc)
ARCANE_ADD_TEST(mesh_sphere_vtk42_binary testMesh-sphere-vtk42-binary.arc)
arcane_add_test_sequential(mesh1_honeycomb2d testMesh-honeycomb2D-1.arc "-We,ARCANE_PRINT_APPLYSIMDPADDING,2")
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
ARCANE_ADD_TEST_SEQUENTIAL(ios ${ARCANE_TEST_PATH}/testIos.arc "-We,ARCANE_USE_PARALLEL_MSH_READER,0")
# Désactive temporairement ce test tant qu'on ne détecte pas correctement la dimension
# du maillage
#arcane_add_test_sequential(ios_msh testIos-msh.arc)
arcane_add_test_sequential(ios_msh2 testIos-msh2.arc)
arcane_add_test_sequential(ios_msh3 testIos-msh3.arc)
arcane_add_test_sequential(ios_msh4 testIos-msh4.arc)
arcane_add_test_sequential(ios_msh5 testIos-msh5.arc "-We,ARCANE_USE_PARALLEL_MSH_READER,0")
arcane_add_test_sequential(ios_msh5_parallel testIos-msh5.arc)
arcane_add_test_sequential(ios_msh6_parallel testIos-msh6.arc "-We,ARCANE_FACE_UNIQUE_ID_BUILDER_VERSION,5")
if (ARCANE_DEFAULT_PARTITIONER_IS_METIS)
  arcane_add_test_parallel_thread(ios_msh4 testIos-msh4.arc 4)
  arcane_add_test_parallel_thread(ios_msh5 testIos-msh5.arc 5 "-We,ARCANE_USE_PARALLEL_MSH_READER,0")
  arcane_add_test_parallel(ios_msh5_parallel testIos-msh5.arc 4)
  arcane_add_test_parallel(ios_msh6_parallel testIos-msh6.arc 4)
  # Ne fonctionne pas encore
  # arcane_add_test_parallel(ios_msh6_parallel_face5 testIos-msh6.arc 4 "-We,ARCANE_FACE_UNIQUE_ID_BUILDER_VERSION,5")
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

arcane_add_test(checkpoint_basic testCheckpoint-3.arc -c 3 -m 5)
arcane_add_test(checkpoint_basic_hash_v2 testCheckpoint-3.arc -c 3 -m 5 "-We,ARCANE_VARIABLEMNG_HASHV2,1" "-We,ARCANE_USE_JSON_METADATA,1")
arcane_add_test(checkpoint_basic2-v1 testCheckpoint-basic2-v1.arc -c 3 -m 5)
arcane_add_test(checkpoint_basic2 testCheckpoint-basic2.arc -c 3 -m 5)
arcane_add_test(checkpoint_basic2-v3 testCheckpoint-basic2-v3.arc -c 3 -m 5)
arcane_add_test(checkpoint_basic2-v3_json_metadata testCheckpoint-basic2-v3.arc -c 3 -m 5 -We,ARCANE_USE_JSON_METADATA,1)
arcane_add_test(checkpoint_basic2-v3_xml_metadata testCheckpoint-basic2-v3.arc -c 3 -m 5 -We,ARCANE_USE_JSON_METADATA,0)
arcane_add_test(checkpoint_basic_hash_file testCheckpoint-basic2-v3.arc -c 3 -m 5 -We,ARCANE_HASHDATABASE_DIRECTORY,${CMAKE_CURRENT_BINARY_DIR}/hashdb)

if (ARCANE_ENABLE_REDIS_TEST)
  arcane_add_test(checkpoint_basic_hash_redis testCheckpoint-basic2-v3.arc -c 3 -m 5 "-We,ARCANE_HASHDATABASE_REDIS,127.0.0.1")
endif()
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
arcane_add_test_sequential(hydro3_cartesian_small testHydro-3-cartesian-small.arc -m 10)
arcane_add_test_parallel_thread(hydro3_cartesian_small testHydro-3-cartesian-small.arc 4 -m 10 "-We,ARCANE_USE_ACTIVE_SHM_QUEUE,1")
arcane_add_test_parallel_thread(hydro3_cartesian_small_gl0_opt testHydro-3-cartesian-small.arc 4 -m 10 "-We,ARCANE_GHOSTLAYER_USE_OPTIMIZED_LAYER,0")
arcane_add_test_parallel_thread(hydro3_cartesian_small_gl2_opt testHydro-3-cartesian-small.arc 4 -m 10 "-We,ARCANE_GHOSTLAYER_USE_OPTIMIZED_LAYER,2")
arcane_add_test_parallel_thread(hydro3_cartesian_small_gl3_opt testHydro-3-cartesian-small.arc 4 -m 10 "-We,ARCANE_GHOSTLAYER_USE_OPTIMIZED_LAYER,3")
arcane_add_test_script(continue_hydro3_cartesian_small continue_hydro3_cartesian_small.xml)
# TODO : À réactiver.
#arcane_add_test_script(continue_hydro3_cartesian_small_4shm continue_hydro3_cartesian_small_4shm.xml)
arcane_add_test_parallel(hydro3_cartesian_8proc testHydro-3-cartesian.arc 8 "-m 50")

arcane_add_test(ArcaneBasicMeshsubdividerservice1-subdivide1 testArcaneBasicMeshSubdividerService1-subdivide1.arc -m 1)
arcane_add_test_parallel_thread(arcanebasicmeshsubdividerservice1-subdivide1 testArcaneBasicMeshSubdividerService1-subdivide1.arc 4 -m 1)
if( Parmetis_FOUND )
  arcane_add_test_parallel_thread(arcanebasicmeshsubdividerservice1-subdivide1-parmetis testArcaneBasicMeshSubdividerService1-subdivide1-parmetis.arc 4 -m 1)
endif()

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

if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_accelerator_test_sequential(hydro_accelerator5 testHydroAccelerator-5.arc -m 50)
  arcane_add_accelerator_test_sequential(hydro_accelerator5_prefetch testHydroAccelerator-5.arc -m 50 "-We,ARCANE_ACCELERATOR_PREFETCH_COMMAND,1")
  arcane_add_accelerator_test_parallel(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_accelerator_test_parallel_thread(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_accelerator_test_message_passing_hybrid(hydro_accelerator5 CASE_FILE testHydroAccelerator-5.arc NB_SHM 2 NB_MPI 2 ARGS -m 50)
  arcane_add_test(hydro_accelerator5 testHydroAccelerator-5.arc -m 50)
  arcane_add_test_sequential(hydro_accelerator5_prefetch testHydroAccelerator-5.arc -m 50 "-We,ARCANE_ACCELERATOR_PREFETCH_COMMAND,1")
  arcane_add_test_sequential_task(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_test_parallel_thread(hydro_accelerator5 testHydroAccelerator-5.arc 4 -m 50)
  arcane_add_test_message_passing_hybrid(hydro_accelerator5 CASE_FILE testHydroAccelerator-5.arc NB_SHM 2 NB_MPI 2 ARGS -m 50)
  if (ARCANE_ACCELERATOR_RUNTIME_NAME STREQUAL cuda)
    arcane_add_accelerator_test_sequential(hydro_accelerator5_cupti testHydroAccelerator-5.arc -m 20 -We,ARCANE_CUPTI_LEVEL,1)
  endif()

  arcane_add_accelerator_test_sequential(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc -m 50)
  arcane_add_test_sequential(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc -m 50)
  arcane_add_test_sequential_task(hydro_accelerator5_multiple_queue testHydroAccelerator-5-multiple_queue.arc 4 -m 50)
endif()

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
    arcane_add_test_sequential(hydro5_papi_init testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Papi -We,ARCANE_PROFILE_INIT,1)
    arcane_add_test_sequential(hydro5_papi_mesh testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Papi -We,ARCANE_PROFILE_CREATE_MESH,1)
    arcane_add_test_sequential_task(hydro5_papi testHydro-5.arc 4 -m 50 -We,ARCANE_PROFILING,Papi)
  endif()
endif()

if (LibUnwind_FOUND)
  arcane_add_test_sequential(hydro5_prof testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Prof)
  arcane_add_test_sequential(hydro5_prof_init testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Prof -We,ARCANE_PROFILE_INIT,1)
  arcane_add_test_sequential(hydro5_prof_mesh testHydro-5.arc -m 50 -We,ARCANE_PROFILING,Prof -We,ARCANE_PROFILE_CREATE_MESH,1)
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
  arcane_add_test_generic(hydro1_vtkhdf CASE_FILE testHydro-1-vtkhdf.arc ARGS "-m 50")
  # Le test suivant ne passe en mode 'mémoire partagée' que si on utilise Metis
  # comme partitionneur par défaut
  if (ARCANE_DEFAULT_PARTITIONER_IS_METIS)
    arcane_add_test_generic(hydro1_vtkhdfv2 CASE_FILE testHydro-1-vtkhdfv2.arc ARGS "-m 50")
  else()
    arcane_add_test_generic(hydro1_vtkhdfv2 CASE_FILE testHydro-1-vtkhdfv2.arc ARGS "-m 50" MP_SEQUENTIAL MP_MPI MP_HYBRID)
  endif()
  arcane_add_test_generic(hydro1_vtkhdfv2_not_collective CASE_FILE testHydro-1-vtkhdfv2-not-collective.arc MP_MPI ARGS "-m 50")
  arcane_add_test_generic(hydro1_vtkhdfv2_backward CASE_FILE testHydro-1-vtkhdfv2-backward.arc ARGS "-m 58")
endif()

if (ARCANE_HAS_KDI)
  arcane_add_test_generic(hydro1_kdi MP_SEQUENTIAL CASE_FILE testHydro-1-kdi.arc ARGS "-m 50")
endif()

ARCANE_ADD_TEST(hydro_depend1 testHydroDepend-1.arc "-m 25")
ARCANE_ADD_TEST(hydro2 testHydro-2.arc "-m 25")

if (HDF5_FOUND)
  ARCANE_ADD_TEST(hydro1_checkpoint testHydro-1-checkpoint.arc -c 2 -m 25)
  ARCANE_ADD_TEST_SEQUENTIAL(hydro6_checkpoint testHydro-6.arc -c 2 -m 25)
  ARCANE_ADD_TEST_PARALLEL(hydro1_checkpoint_3proc_rep4 testHydro-1-checkpoint.arc 12 -c 2 -m 25 -R 4)
endif()

macro(arcane_add_accelerator_test_all_policies TEST_NAME CASE_FILE)
  arcane_add_test_sequential(${TEST_NAME} ${CASE_FILE})
  arcane_add_test_sequential_task(${TEST_NAME} ${CASE_FILE} 4)
  arcane_add_accelerator_test_sequential(${TEST_NAME} ${CASE_FILE})
endmacro()

if (ARCANE_HAS_ACCELERATOR_API)
  arcane_add_test_sequential(numarray1 testNumArray-1.arc)
  arcane_add_test_sequential_task(numarray1 testNumArray-1.arc 4)
  arcane_add_accelerator_test_sequential(numarray1 testNumArray-1.arc)

  arcane_add_test_sequential(runqueue1 testRunQueue-1.arc)
  arcane_add_test_sequential_task(runqueue1 testRunQueue-1.arc 4)
  arcane_add_accelerator_test_sequential(runqueue1 testRunQueue-1.arc)

  arcane_add_test_sequential(acceleratorviews1 testAcceleratorViews-1.arc)
  arcane_add_test_sequential_task(acceleratorviews1 testAcceleratorViews-1.arc 4)
  arcane_add_accelerator_test_sequential(acceleratorviews1 testAcceleratorViews-1.arc)
  arcane_add_accelerator_test_parallel(acceleratorviews1 testAcceleratorViews-1.arc 4)

  arcane_add_test_sequential(accelerator_reduce1 testAcceleratorReduce-1.arc)
  arcane_add_test_sequential_task(accelerator_reduce1 testAcceleratorReduce-1.arc 4)
  arcane_add_accelerator_test_sequential(accelelerator_reduce1 testAcceleratorReduce-1.arc)
  arcane_add_accelerator_test_sequential(accelelerator_reduce1_atomic testAcceleratorReduce-1-atomic.arc)

  arcane_add_test_sequential(accelerator_scan1 testAcceleratorScan-1.arc)
  arcane_add_test_sequential_task(accelerator_scan1 testAcceleratorScan-1.arc 4)
  arcane_add_accelerator_test_sequential(accelelerator_scan1 testAcceleratorScan-1.arc)

  arcane_add_test_sequential(accelerator_filter1 testAcceleratorFilter-1.arc)
  arcane_add_test_sequential_task(accelerator_filter1 testAcceleratorFilter-1.arc 4)
  arcane_add_accelerator_test_sequential(accelelerator_filter1 testAcceleratorFilter-1.arc)

  arcane_add_test_sequential(accelerator_partitioner1 testAcceleratorPartitioner-1.arc)
  arcane_add_test_sequential_task(accelerator_partitioner1 testAcceleratorPartitioner-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_partitioner1 testAcceleratorPartitioner-1.arc)

  arcane_add_test_sequential(accelerator_sorter1 testAcceleratorSorter-1.arc)
  arcane_add_test_sequential_task(accelerator_sorter1 testAcceleratorSorter-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_sorter1 testAcceleratorSorter-1.arc)

  arcane_add_test_sequential(accelerator_material1 testAcceleratorMaterials-1.arc)
  arcane_add_test_sequential_task(accelerator_material1 testAcceleratorMaterials-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_material1 testAcceleratorMaterials-1.arc)

  arcane_add_test_sequential(accelerator_material1_old testAcceleratorMaterials-1.arc "-We,ARCANE_MATERIALMNG_USE_ACCELERATOR_FOR_CONSTITUENTITEMVECTOR,0")
  arcane_add_accelerator_test_sequential(accelerator_material1_old testAcceleratorMaterials-1.arc "-We,ARCANE_MATERIALMNG_USE_ACCELERATOR_FOR_CONSTITUENTITEMVECTOR,0")

  arcane_add_test_sequential(memorycopy1 testMemoryCopy-1.arc)
  arcane_add_test_sequential_task(memorycopy1 testMemoryCopy-1.arc 4)
  arcane_add_accelerator_test_sequential(memorycopy1 testMemoryCopy-1.arc)

  arcane_add_test_sequential(multi_memorycopy1 testMultiMemoryCopy-1.arc)
  arcane_add_test_sequential_task(multi_memorycopy1 testMultiMemoryCopy-1.arc 4)
  arcane_add_accelerator_test_sequential(multi_memorycopy1 testMultiMemoryCopy-1.arc)

  arcane_add_test_sequential(accelerator_iteminfo1 testAcceleratorItemInfo-1.arc)
  arcane_add_test_sequential_task(accelerator_iteminfo1 testAcceleratorItemInfo-1.arc 4)
  arcane_add_accelerator_test_sequential(accelerator_iteminfo1 testAcceleratorItemInfo-1.arc)

  arcane_add_test_sequential(miniweather_orig1 testMiniWeatherOriginal.arc)
  arcane_add_test_sequential(miniweather_array_seq1 testMiniWeatherArraySequential.arc)
  arcane_add_test_sequential(miniweather_array1_right_layout testMiniWeatherArray.arc)
  arcane_add_test_sequential(miniweather_array1_left_layout testMiniWeatherArrayLeftLayout.arc)
  arcane_add_test_sequential_task(miniweather_array1 testMiniWeatherArray.arc 4)
  arcane_add_accelerator_test_sequential(miniweather_array1_right_layout testMiniWeatherArray.arc)
  arcane_add_accelerator_test_sequential(miniweather_array1_left_layout testMiniWeatherArrayLeftLayout.arc)
  arcane_add_accelerator_test_sequential(miniweather_array1_device testMiniWeatherArrayDevice.arc)

  arcane_add_accelerator_test_all_policies(accelerator_atomic testAcceleratorAtomic-1.arc)

  arcane_add_test_sequential(acceleratormath1 testAcceleratorMath-1.arc)
  arcane_add_accelerator_test_sequential(acceleratormath1 testAcceleratorMath-1.arc)
endif()

arcane_add_test_sequential(pdes_random_number_generator_service testPDESRandomNumberGenerator.arc)
arcane_add_test_sequential(simple_csv_output_service testSimpleCsvOutputService.arc)
arcane_add_test_sequential(simple_csv_comparator_service testSimpleCsvComparatorService.arc)

arcane_add_test(geometry testGeom.arc)
arcane_add_test(geometry_applyoperation_v2 testGeom.arc -We,ARCANE_DEBUG_APPLYOPERATION,1)
arcane_add_test(geometry_applyoperation_v1 testGeom.arc -We,ARCANE_APPLYOPERATION_VERSION,1 -We,ARCANE_DEBUG_APPLYOPERATION,1)

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
  arcane_add_csharp_test_sequential(hydro_cs testSimpleHydroCS-1.arc -m 15 "-We,ARCANE_DOTNET_USE_LEGACY_DESTROY,0")
  if (ARCANE_HAS_TASKS)
    arcane_add_csharp_test_sequential(hydro_cs_task testSimpleHydroCS-1.arc -m 15 -K 4)
  endif()
  arcane_add_csharp_test_sequential(casefunction_cs1 testCaseFunctionModule-1.arc "-We,ARCANE_DOTNET_USE_LEGACY_DESTROY,0")
  arcane_add_csharp_test_sequential_external_assembly(casefunction_cs2 testCaseFunctionModule-2.arc.in ExternalDotNetFunction1)
endif()

arcane_add_test_parallel(loadbalance_test1 testLoadBalanceHydro-MeshPartitionerTester.arc 4 -m 30)
arcane_add_test_parallel(loadbalance_test1_d2 testLoadBalanceHydro-MeshPartitionerTester2.arc 4 -m 30)
arcane_add_test_parallel(loadbalance_test1_checkpoint testLoadBalanceHydro-MeshPartitionerTester3.arc 4 -c 3 -m 15)
arcane_add_test_parallel(loadbalance_test1 testLoadBalanceHydro-MeshPartitionerTester.arc 12 -m 30)
arcane_add_test_parallel(loadbalance_test1_3exchange_v1 testLoadBalanceHydro-MeshPartitionerTester.arc 6 -We,ARCANE_NB_EXCHANGE=3 -m 30)
arcane_add_test_parallel(loadbalance_test1_3exchange_v2 testLoadBalanceHydro-MeshPartitionerTester.arc 6 -We,ARCANE_NB_EXCHANGE,3 -We,ARCANE_MESH_EXCHANGE_VERSION,2 -m 30)
if(Parmetis_FOUND)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_metis1 testLoadBalanceHydro-Metis.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL_THREAD(loadbalance_metis3 testLoadBalanceHydro-Metis3.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_metis3 testLoadBalanceHydro-Metis3.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_metis1_no_node testLoadBalanceHydro-Metis2.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_metis1_rep3 testLoadBalanceHydro-Metis.arc 12 -m 30 -R 3)
endif()
if(Zoltan_FOUND)
  ARCANE_ADD_TEST_PARALLEL(zoltan1 testPartition-zoltan.arc 4)
  ARCANE_ADD_TEST_PARALLEL(zoltan1_legacy_partition_init testPartition-zoltan.arc 4 "-We,ARCANE_USE_LEGACY_INIT_LOADBALANCEMNG,1")
  ARCANE_ADD_TEST_PARALLEL(loadbalance_zoltanH testLoadBalanceHydro-Zoltan-hypergraphe.arc 4 -m 30)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_zoltanG testLoadBalanceHydro-Zoltan-geometric.arc 4 -m 30)
endif()
if(PTScotch_FOUND)
  arcane_add_test_parallel(ptscotch1_legacy_partition_init testPartition-ptscotch.arc 4 "-We,ARCANE_USE_LEGACY_INIT_LOADBALANCEMNG,1")
  ARCANE_ADD_TEST_PARALLEL(ptscotch1 testPartition-ptscotch.arc 4)
  ARCANE_ADD_TEST_PARALLEL(loadbalance_ptscotch1 testLoadBalanceHydro-PTScotch.arc 4 -m 80)
endif()

ARCANE_ADD_TEST_PARALLEL_THREAD(thread1 testThread-1.arc 1 -m 15)
arcane_add_test_sequential(task1_0 testTask-1.arc 1 -m 5)
arcane_add_test_sequential_task(task1 testTask-1.arc 1 -m 5)
arcane_add_test_sequential_task(task1 testTask-1.arc 4 -m 5)
arcane_add_test_sequential_task(task1_cppstd testTask-1.arc 4 -m 5 -A,ThreadService=Cpp)
arcane_add_test_sequential_task(task1_glib testTask-1.arc 4 -m 5 -A,ThreadService=Glib)
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
arcane_add_test_script(compare_seq_seq_hash_only compare_seq_seq_hash_only.xml)
arcane_add_test_script(compare_seq_par_hash_only compare_seq_par_hash_only.xml)
arcane_add_test_script(compare_par_par_hash_only compare_par_par_hash_only.xml)
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

#################################################################
#################################################################

# Test pour la partie StringVariableReplace.
arcane_add_test_sequential(string_variable_replace testStringVariableReplace.arc)

#################################################################
#################################################################

# Test pour les implementations de TimeHistoryAdder.
arcane_add_test(time_history_adder_1 testTimeHistoryAdder-1.arc -c 2 -m 5 -We,ARCANE_ENABLE_NON_IO_MASTER_CURVES,1)

#################################################################
function(arcane_add_python_test_sequential test_name script_name case_file)
  arcane_add_test_sequential(${test_name} "${ARCANE_TEST_PATH}/testPython-${script_name}.arc" -A,UsingDotNet=1 ${ARGN})
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PYTHONNET_PYDLL=${Python_LIBRARIES}")
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python")
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${Python_SITEARCH}")
  set(ARCANE_TEST_PYTHON_SCRIPT ${script_name})
  configure_file(${ARCANE_ARC_PATH}/${case_file}.in "${ARCANE_TEST_PATH}/testPython-${script_name}.arc")
endfunction()

#################################################################
# Test wrapper python en mode embedded
if (ARCANE_ENABLE_DOTNET_PYTHON_WRAPPER)
  # Il faut trouver le chemin de 'libpython.so' et positionner
  # la variable d'environnement PYTHONNET_PYDLL pour le mode embedded.
  # TODO: Positionner directement lors de la compilation la propriété C# Python.Runtime.PythonDLL
  set(Python_USE_STATIC_LIBS FALSE)
  find_package(Python COMPONENTS Interpreter Development NumPy)
  message(STATUS "Python FOUND=${Python_FOUND}")

  if (Python_FOUND)
    message(STATUS "Python NumPy FOUND=${Python_NumPy_FOUND}")
    message(STATUS "PythonLibraries=${Python_LIBRARIES}")
    message(STATUS "Python Python_RUNTIME_LIBRARY_DIRS=${Python_RUNTIME_LIBRARY_DIRS}")
    message(STATUS "Python_SITEARCH=${Python_SITEARCH}")
    arcane_add_python_test_sequential(python_test_embed1 script1 "testExternalPlugin-1.arc")
    arcane_add_python_test_sequential(python_test_embed2 script2 "testExternalPlugin-2.arc")
    arcane_add_python_test_sequential(python_test_embed3 script3 "testExternalPlugin-3.arc")
    arcane_add_python_test_sequential(python_test_embed4 script4 "testExternalPlugin-4.arc")
  endif()
  #arcane_add_test_sequential(python_wrapping_test1 testSimplePythonCall-1.arc -Z --dotnet-runtime=coreclr -m 5)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
