# Il faut que swig soit disponible
find_package(ArcaneSwig
  HINTS "${Arcane_DIR}"
  )
if (NOT ArcaneSwig_FOUND)
  message(STATUS "INFO: Disabling 'C#' samples because 'ArcaneSwig' package is not available. ")
  return()
endif()
if (NOT ARCANE_CORECLR_VERSION STREQUAL "6.0")
  message(STATUS "INFO: Disabling 'C#' samples because '.Net' version is not '6.0' (version=${ARCANE_CORECLR_VERSION})")
  return()
endif()

# ----------------------------------------------------------------------------
# Répertoire contenant les sources du wrapper
add_subdirectory(wrapper)

# Répertoire contenant une implémentation d'EOS utilisant un gaz parfait
add_subdirectory(perfectgas)

# Répertoire contenant une implémentation du main en C# (optionel)
add_subdirectory(main)

# ----------------------------------------------------------------------------
# Ajoute test
configure_file(EOS.csharp.arc ${EOS_BINARY_DIR} @ONLY)
set(ASSEMBLY_ARGS "-A,DotNetMainAssemblyName=${EOS_BINARY_DIR}/PerfectGas.dll")

# 'mono' en mode embarqué
add_test(NAME eos-cs-mono
  COMMAND ${EOS_BINARY_DIR}/EOS -A,DotNetEmbeddedRuntime=mono ${ASSEMBLY_ARGS} EOS.csharp.arc
  WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )
# 'coreclr' en mode embarqué
add_test(NAME eos-cs-coreclr
  COMMAND ${EOS_BINARY_DIR}/EOS -A,DotNetEmbeddedRuntime=coreclr ${ASSEMBLY_ARGS} EOS.csharp.arc
  WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )

if (ARCANE_DOTNET_HAS_RUNTIME_coreclr)
  # 'coreclr' en mode direct
  add_test(NAME eos-cs-coreclr-direct
    COMMAND ${ARCANE_DOTNET_RUNTIME_coreclr} ${EOS_BINARY_DIR}/EOS.Main.dll ${ASSEMBLY_ARGS} EOS.csharp.arc
    WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )
  # TODO: a partir de CMake 3.22, on peut utiliser ENVIRONMENT_MODIFICATION qui permet d'ajouter des
  # valeurs à une variable existante. Il faut utiliser cela
  # TODO: gérer test Win32.
  set_tests_properties(eos-cs-coreclr-direct PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${ARCANE_PREFIX_DIR}/lib")
endif()
if (ARCANE_DOTNET_HAS_RUNTIME_mono)
  # 'mono' en mode direct
  add_test(NAME eos-cs-mono-direct
    COMMAND ${ARCANE_DOTNET_RUNTIME_mono} ${EOS_BINARY_DIR}/EOS.Main.dll ${ASSEMBLY_ARGS} EOS.csharp.arc
    WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )
  set_tests_properties(eos-cs-mono-direct PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${ARCANE_PREFIX_DIR}/lib")
endif()

