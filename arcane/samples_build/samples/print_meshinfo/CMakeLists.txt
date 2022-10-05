project(PrintMeshInfo LANGUAGES C CXX)

find_package(Arcane)
add_executable(PrintMeshInfo main.cc)
arcane_add_arcane_libraries_to_target(PrintMeshInfo)

# Ajoute test d'affichage d'un maillage
configure_file(sod.vtk ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(sod.vtkfaces.vtk ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
add_test(NAME print_meshinfo1 COMMAND PrintMeshInfo sod.vtk)
