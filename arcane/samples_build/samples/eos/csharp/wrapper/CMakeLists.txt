# ----------------------------------------------------------------------------
# TODO: séparer la génération du wrapper de son utilisation (faire un projet contenant uniquement EOSPerfectGas.cs)
set_property(SOURCE EOSCSharp.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE EOSCSharp.i PROPERTY INCLUDE_DIRECTORIES ${EOS_SOURCE_DIR} ${ARCANE_DOTNET_WRAPPER_INCLUDE_DIRS})
set_property(SOURCE EOSCSharp.i PROPERTY COMPILE_OPTIONS -namespace EOS)
set(EOS_CSHARP_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${EOS_CSHARP_OUTDIR})

# NOTE: le nom doit commencer par 'lib' pour que mono trouve la bibliothèque
# au moment de l'exécution
swig_add_library(libeos_csharp
  TYPE SHARED
  LANGUAGE CSHARP
  SOURCES EOSCSharp.i
  OUTPUT_DIR ${EOS_CSHARP_OUTDIR}/cs_files 
  OUTFILE_DIR ${EOS_CSHARP_OUTDIR}/cpp_files
  )
target_link_libraries(libeos_csharp PUBLIC arcane_dotnet_wrapper_core)
target_include_directories(libeos_csharp PUBLIC ${EOS_SOURCE_DIR})
# Il faut que la cible soit installée au même endroit que l'exécutable de test pour
# qu'elle soit trouvée facilement.
set_property(TARGET libeos_csharp PROPERTY LIBRARY_OUTPUT_DIRECTORY ${EOS_BINARY_DIR})

configure_file(EOS.csproj.in ${CMAKE_CURRENT_BINARY_DIR}/EOS.csproj @ONLY)

# Ajoute une commande et une cible pour compiler le C# généré par 'swig'
# (la fonction ci-dessous est définie dans samples.utils.cmake)
arcane_sample_add_csharp_target(TARGET_NAME dotnet_eos_csharp_wrapper
  PROJECT_NAME EOS
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  PUBLISH_DIRECTORY ${EOS_BINARY_DIR}
  DEPENDS libeos_csharp
  )

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
