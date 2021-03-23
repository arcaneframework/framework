# Il faut que swig soit disponible
find_package(ArcaneSwig
  HINTS "${Arcane_DIR}"
  )
if (NOT ArcaneSwig_FOUND)
  message(STATUS "Can not find 'ArcaneSwig' for EOS C# sample. Disabling 'C#' samples")
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

# 'mono' en mode embarqué
add_test(NAME eos-cs-mono
  COMMAND ${EOS_BINARY_DIR}/EOS -A,DotNetEmbeddedRuntime=mono EOS.csharp.arc
  WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )
# 'coreclr' en mode embarqué
add_test(NAME eos-cs-coreclr
  COMMAND ${EOS_BINARY_DIR}/EOS -A,DotNetEmbeddedRuntime=coreclr EOS.csharp.arc
  WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )

if (ARCANE_DOTNET_HAS_RUNTIME_coreclr)
  # 'coreclr' en mode direct
  add_test(NAME eos-cs-coreclr-direct
    COMMAND ${ARCANE_DOTNET_RUNTIME_coreclr} ${EOS_BINARY_DIR}/EOS.Main.dll -A,DotNetMainAssemblyName=${EOS_BINARY_DIR}/PerfectGas.dll EOS.csharp.arc
    WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )
endif()
if (ARCANE_DOTNET_HAS_RUNTIME_mono)
  # 'mono' en mode direct
  add_test(NAME eos-cs-mono-direct
    COMMAND ${ARCANE_DOTNET_RUNTIME_mono} ${EOS_BINARY_DIR}/EOS.Main.dll -A,DotNetMainAssemblyName=${EOS_BINARY_DIR}/PerfectGas.dll EOS.csharp.arc
    WORKING_DIRECTORY ${EOS_BINARY_DIR}
  )
endif()

