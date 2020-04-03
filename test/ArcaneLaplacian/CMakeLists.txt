createExecutable(laplacian)

addSources(laplacian
        main.cc
        ArcaneCodeService.cc
        LaplacianModule.cc
        )

arcane_target_add_axl(laplacian Laplacian)
#generateAxl(laplacian
#  Laplacian.axl
#  NO_COPY
#  )

if (TARGET intel)
    linkLibraries(laplacian intel)
endif ()

linkLibraries(laplacian
        Alien::alien_core
        arcane_core
        arcane_std
        arcane_mpi
        arcane_impl
        arcane_mesh
        arcane_utils
        arcane_thread
        )

commit(laplacian)

configure_file(Laplacian.config ${CMAKE_CURRENT_BINARY_DIR}/Laplacian.config COPYONLY)
configure_file(Laplacian.arc ${CMAKE_CURRENT_BINARY_DIR}/Laplacian.arc COPYONLY)

# Copie le fichier de maillage dans le repertoire courant
#CONFIGURE_FILE(${ARCANESRCROOT}/maillages/tube5x5x100.vtk ${ARCANE_FULL_SAMPLE_DIR}/sod.vtk COPYONLY)
#CONFIGURE_FILE(${ARCANESRCROOT}/maillages/tube5x5x100.vtkfaces.vtk ${ARCANE_FULL_SAMPLE_DIR}/sod.vtkfaces.vtk COPYONLY)
