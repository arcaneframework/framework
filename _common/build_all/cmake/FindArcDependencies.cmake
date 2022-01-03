if (NOT ARCDEPENDENCIES_ROOT)
  message(FATAL_ERROR "Variable 'ARCDEPENDENCIES_ROOT' is not set")
endif()

# Vérifie que le répertoire 'dependencies' n'est pas vide.
# Si c'est le cas cela signifie qu'on n'a pas cloner les sous-modules
if (EXISTS "${ARCDEPENDENCIES_ROOT}/CMakeLists.txt")
  add_subdirectory(${ARCDEPENDENCIES_ROOT} dependencies)
else()
  message(FATAL_ERROR
    "Directory 'dependencies' is empty. "
    "You need to clone this git submodule with the following command:\n"
    "  cd ${CMAKE_SOURCE_DIR}\n"
    "  git submodule update --init\n"
    )
endif()

set(ArcDependencies_FOUND YES)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
