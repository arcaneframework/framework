# NOTE: Python binding is a work in progress. DO NOT USE IT

set(ARCANE_DOTNET_DLLS_DIR "${CMAKE_BINARY_DIR}/lib")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ArcanePython")
configure_file(ArcanePython/__init__.py.in ArcanePython/__init__.py @ONLY)
configure_file(ArcanePython/app.runtime.json ArcanePython COPYONLY)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ArcanePython/__init__.py"
  ArcanePython/app.runtime.json
  DESTINATION "ArcanePython")
install(DIRECTORY tests DESTINATION . USE_SOURCE_PERMISSIONS)
enable_testing()

# Crée un script pour lancer python avec le bon pythonpath
# TODO: Faire windows
configure_file(python_wrapper.sh.in python_wrapper.sh @ONLY)

add_test(NAME python_test1 COMMAND ./python_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test1.py)
add_test(NAME python_test4 COMMAND ./python_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test4.py)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
