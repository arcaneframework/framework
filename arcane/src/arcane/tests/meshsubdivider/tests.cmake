set(ARCANE_TEST_CASEPATH ${TEST_PATH}/meshsubdivider)


set(subdivider_write_output_file true)

set(output_file_arg "-A,//mesh-subdivider-tester/test/output-file=")

# ArcaneBasicMeshSubdivider

# old
#arcane_add_test(basicmeshsubdividerservice1-subdivide1 testArcaneBasicMeshSubdividerService1-subdivide1.arc -m 1)
#arcane_add_test_parallel_thread(basicmeshsubdividerservice1-subdivide1 testArcaneBasicMeshSubdividerService1-subdivide1.arc 4 -m 1)
#arcane_add_test_sequential(basicmeshsubdividerservidetriangle-subdivide-1 testArcaneBasicMeshSubdividerServiceTriangle.arc -m 1 "-A,//meshes/mesh/subdivider/different-element-type-output=1")
#arcane_add_test(basicmeshsubdividerservidetriangle-subdivide testArcaneBasicMeshSubdividerServiceTriangle.arc -m 1)
#arcane_add_test(basicmeshsubdividerservidequadrilateral-subdivide testArcaneBasicMeshSubdividerServiceQuadrilateral.arc -m 1)
#arcane_add_test_sequential(basicmeshsubdividerservidetriangle-subdivide-1 testArcaneBasicMeshSubdividerServiceTriangle.arc -m 1 "-A,//meshes/mesh/subdivider/different-element-type-output=1")

# new
# Copie pour ajouter dans le build
#file(COPY ${TEST_PATH}/ArcaneBasicMeshSubdividerTestData DESTINATION ${ARCANE_TEST_WORKDIR} USE_SOURCE_PERMISSIONS)
# arcane_copy_mesh_direct(one_hex.vtk)
# arcane_copy_mesh_direct(one_quad.vtk)
# arcane_copy_mesh_direct(one_tet.vtk)
# arcane_copy_mesh_direct(one_tri.vtk)

# just to test the ... test
arcane_add_test(meshservice-subdivide1-hydro testHydro-3-meshservice-subdivide1.arc)


# Using Zedono and GitAnnexe
# Setup command tests tri_quad tri_tri and generate test with nb subdivision in 1 2 3.
# Generate one vtk file as result output_
foreach (nb_element IN ITEMS "one" "two" )
  foreach(tuple IN ITEMS "tri;quad" "quad;tri" "tet;hex" "hex;tet")
    list(GET tuple 0 src)
    list(GET tuple 1 dst)
    set(mesh_arg "-A,//meshes/mesh/filename=${ARCANE_TEST_PATH}/mesh_subdivider/${nb_element}_${src}.vtk")

    foreach(nbsub RANGE 1 3)
      set(nb_subdivision_arg "-A,//meshes/mesh/subdivider/nb-subdivision=${nbsub}")
      set(with_same_pattern "-A,//meshes/mesh/subdivider/different-element-type-output=0")
      set(with_different_pattern "-A,//meshes/mesh/subdivider/different-element-type-output=1")
      # src -> src
      arcane_add_test_sequential(basicmeshsubdividerservice_${nb_element}_${src}_to_${src}_subdiv${nbsub}
              testArcaneBasicMeshSubdividerService.arc -m 1
              ${mesh_arg}
              ${nb_subdivision_arg}
              ${with_same_pattern}
              "-A,//unit-test-module/test/output-file=output_${nb_element}_${src}_to_${src}_subdiv${nbsub}")
      # src -> dst
      arcane_add_test_sequential(basicmeshsubdividerservice_${nb_element}_${src}_to_${dst}_subdiv${nbsub}
              testArcaneBasicMeshSubdividerService.arc -m 1
              ${mesh_arg}
              ${nb_subdivision_arg}
              ${with_different_pattern}
              "-A,//unit-test-module/test/output-file=output_${nb_element}_${src}_to_${dst}_subdiv${nbsub}")
    endforeach()
  endforeach ()
endforeach()

set(BIG_TET_MESH_FILES
        #"${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_spe11c.vtk"
        #"${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere_cut.vtk"
        #"${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere.vtk"
        "${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere_no_element.vtk"
        #"${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere_msh.msh"
        "${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere_test.msh"
)
set(BIG_HEX_MESH_FILES
        "${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/hex/hex_ring.vtk"
)

# bigger

set(PARTITIONNERS MeshPartitionerTester)
if(Parmetis_FOUND)
  list(APPEND PARTITIONNERS Metis)
endif()
set(NB_PROCS 1 2 4 8 16)
set(DIFFERENT_ELEMENT_TYPE_OUTPUT 0 1 )
set(NB_SUBDIVISION 0 1 2 3 4 5 6)


foreach ( mesh IN LISTS BIG_TET_MESH_FILES )
  #message("#debug" ${mesh})
  get_filename_component(mesh_name "${mesh}" NAME_WE)
  set(mesh_arg "-A,//meshes/mesh/filename=${mesh}")
  arcane_add_test_sequential(basicmeshsubdividerservice_tet_bigger_subdiv${nbsub}_${mesh_name}_to_tet
          testArcaneBasicMeshSubdividerService.arc -m 1
          "${mesh_arg}"
          "-A,//meshes/mesh/subdivider/nb-subdivision=${nbsub}"
          "-A,//unit-test-module/test/output-file=output_${nb_element}_${src}_to_${src}_subdiv${nbsub}_to_tet")

  arcane_add_test_sequential(basicmeshsubdividerservice_tet_bigger_subdiv${nbsub}_${mesh_name}_to_hex
          testArcaneBasicMeshSubdividerService.arc -m 1
          "${mesh_arg}"
          "-A,//meshes/mesh/subdivider/nb-subdivision=${nbsub}"
          "-A,//meshes/mesh/subdivider/different-element-type-output=1"
          "-A,//unit-test-module/test/output-file=output_${nb_element}_${src}_to_${src}_subdiv${nbsub}_to_hex")
endforeach ()

set(nb_proc 16)
# nb_subdivision = 0 -> pas de renumérotation
# ajouter <subdivider-renumbering>-1 | 0 | 1<subdivider-renumbering>



foreach ( mesh IN LISTS BIG_HEX_MESH_FILES )
  #message("#debug" ${mesh})
  get_filename_component(mesh_name "${mesh}" NAME_WE)
  set(mesh_file "-A,//meshes/mesh/filename=${mesh}")
  set(nb_subdivide "-A,//meshes/mesh/subdivider/nb-subdivision=1")
  set(different_element_type_output "-A,//meshes/mesh/subdivider/different-element-type-output=0")
  arcane_add_test_sequential(basicmeshsubdividerservice_hex_bigger_subdiv${nbsub}_${mesh_name}_to_hex
          testArcaneBasicMeshSubdividerService.arc -m 1
          "${mesh_file}"
          "-A,//meshes/mesh/subdivider/nb-subdivision=${nbsub}"
          "-A,//unit-test-module/test/output-file=output_${nb_element}_${mesh_name}_subdiv${nbsub}_to_hex")

  arcane_add_test_sequential(basicmeshsubdividerservice_hex_bigger_subdiv${nbsub}_${mesh_name}_to_tet
          testArcaneBasicMeshSubdividerService.arc -m 1
          "${mesh_file}"
          "-A,//meshes/mesh/subdivider/nb-subdivision=${nbsub}"
          "-A,//meshes/mesh/subdivider/different-element-type-output=1"
          "-A,//unit-test-module/test/output-file=output_${nb_element}_${mesh_name}_subdiv${nbsub}_to_tet")

  if (Parmetis_FOUND)
#    arcane_add_test_parallel_thread(basicmeshsubdividerservice_bigger_hex_bigger_subdiv${nbsub}_${mesh_name}_to_hex_thread testArcaneBasicMeshSubdividerService1-subdivide1-parmetis.arc 4 -m 2)
#    arcane_add_test_parallel(basicmeshsubdividerservice_bigger_hex_to_hex_bigger_subdiv_mpi testArcaneBasicMeshSubdividerService1-subdivide1-parmetis.arc ${nb_proc} ${mesh_file} ${nb_subdivide} )

    arcane_add_test_parallel(basicmeshsubdividerservice_bigger_hex_to_tet_subdiv_mpi testArcaneBasicMeshSubdividerService.arc ${nb_proc} "-A,//meshes/mesh/filename=${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere_test.msh" ${nb_subdivide} ${different_element_type_output} -m 1 )
    #arcane_add_test_parallel(${test_name} ${ARGS_CASE_FILE} ${ARGS_NB_MPI} ${ARGS_ARGS})
  endif ()
endforeach ()


#set( partitioner_variation "Metis" "MeshPartitionerTester")
set( nb_process_variation "16")
set( nb_subdivision_variation "0" "1" "2" ) #"2" "3" )
set( different_element_output_variation "0" "1" )

set( nb_subdivision "-A,//meshes/mesh/subdivider/nb-subdivision")
set( different_element_output "-A,//meshes/mesh/subdivider/different-element-type-output")
set( mesh_file_opt "-A,//meshes/mesh/filename")

set( big_mesh_files
        "${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/hex/hex_ring.vtk"
        "${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere_test.msh"
        "${ARCANE_TEST_PATH}/mesh_subdivider/NumpexSamples/tet/tet_sphere_no_element.vtk"
)

if (Parmetis_FOUND)
  foreach ( mesh_file IN LISTS big_mesh_files )
    foreach (nb_process IN LISTS nb_process_variation )
      foreach ( nb_sub IN LISTS nb_subdivision_variation )
        foreach (deov IN LISTS different_element_output_variation )
          set(nb_subdivision_arg ${nb_subdivision}=${nb_sub})
          set(different_element_output_arg ${different_element_output}=${deov})
          set(mesh_file_arg ${mesh_file_opt}=${mesh_file})
          #message("Variation different_element_output_arg" ${different_element_output_arg})
          #message("Variation nb_subdivision_arg" ${nb_subdivision_arg})
          #message("Variation mesh_file_variation" ${mesh_file_arg})
          #message("Variation nb_process_variation" ${nb_process})
          get_filename_component(mesh_name "${mesh_file_arg}" NAME_WE)
          #message("Variation mesh_name" ${mesh_name})
          arcane_add_test_parallel(basicmeshsubdividerservice_bigger_${mesh_name}_nb_subdivision_${nb_sub}_different_element_type_output_${deov} testArcaneBasicMeshSubdividerService.arc ${nb_process} ${mesh_file_arg} ${nb_subdivision_arg} ${different_element_output_arg} -m 1 )
          # For threads:
          #arcane_add_test_parallel_thread(basicmeshsubdividerservice_bigger_${mesh_name}_nb_subdivision_${nb_sub}_different_element_type_output_${deov} testArcaneBasicMeshSubdividerServiceTriangle.arc ${nb_process} ${mesh_file_arg} ${nb_subdivision_arg} ${different_element_output_arg} -m 1 )
        endforeach ()
      endforeach ()
    endforeach ()
  endforeach ()
endif ()
# ArcaneBasicMeshSubdivider

