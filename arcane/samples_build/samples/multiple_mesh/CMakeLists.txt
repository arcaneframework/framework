project(MultipleMesh LANGUAGES C CXX)

find_package(Arcane REQUIRED)
add_executable(MultipleMesh
  main.cc MultipleMeshModule.cc MultipleMesh_axl.h MeshInfoPrinter_axl.h
)

arcane_generate_axl(MultipleMesh)
arcane_generate_axl(MeshInfoPrinter)

arcane_add_arcane_libraries_to_target(MultipleMesh)
target_include_directories(MultipleMesh PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Recopie le jeu de données et les maillages
configure_file(multiple_mesh_legacy.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(multiple_mesh.arc ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(plancher.msh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(square_v41.msh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(MultipleMesh.config ${CMAKE_CURRENT_BINARY_DIR} @ONLY)

add_test(NAME multiple_mesh_legacy COMMAND MultipleMesh multiple_mesh_legacy.arc)
add_test(NAME multiple_mesh COMMAND MultipleMesh multiple_mesh.arc)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
