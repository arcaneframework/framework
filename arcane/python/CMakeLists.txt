﻿# NOTE: Python binding is a work in progress. DO NOT USE IT

set(ARCANE_DOTNET_DLLS_DIR "${CMAKE_BINARY_DIR}/lib")
set(ARCANE_PYTHON_BASE_INSTALL_DIR "${ARCANE_DOTNET_DLLS_DIR}/python")
set(ARCANE_PYTHON_SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}")
file(MAKE_DIRECTORY "${ARCANE_PYTHON_BASE_INSTALL_DIR}/ArcanePython")
configure_file(ArcanePython/__init__.py.in "${ARCANE_PYTHON_BASE_INSTALL_DIR}/ArcanePython/__init__.py" @ONLY)

set(ARCANE_PYTHON_SRCS
  _utils.py
  SubDomainContext.py
  app.runtime.json)

configure_file(ArcanePython/app.runtime.json "${ARCANE_PYTHON_BASE_INSTALL_DIR}/ArcanePython" COPYONLY)
foreach(file ${ARCANE_PYTHON_SRCS})
  #configure_file(ArcanePython/${file} "${ARCANE_PYTHON_BASE_INSTALL_DIR}/ArcanePython" COPYONLY)
  install(FILES "${ARCANE_PYTHON_BASE_INSTALL_DIR}/ArcanePython/${file}"
    DESTINATION "lib/python/ArcanePython"
  )
endforeach()

install(FILES
  "${ARCANE_PYTHON_BASE_INSTALL_DIR}/ArcanePython/__init__.py"
#  "${ARCANE_PYTHON_BASE_INSTALL_DIR}/ArcanePython/_utils.py"
#  ArcanePython/app.runtime.json
  DESTINATION "lib/python/ArcanePython")
install(DIRECTORY tests DESTINATION . USE_SOURCE_PERMISSIONS)

enable_testing()

# Crée un script pour lancer python avec le bon pythonpath
# TODO: Faire windows
configure_file(python_wrapper.sh.in python_wrapper.sh @ONLY)

add_test(NAME python_test1 COMMAND ./python_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test1.py)
add_test(NAME python_test4 COMMAND ./python_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test4.py)
add_test(NAME python_test5 COMMAND ./python_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test5.py)
add_test(NAME python_test_microhydro
  COMMAND ./python_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_microhydro.py -A,MaxIteration=15 ${CMAKE_CURRENT_SOURCE_DIR}/tests/t3.arc)
add_test(NAME python_test_microhydro_post_processing
  COMMAND ./python_wrapper.sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_microhydro_post_processing.py -A,MaxIteration=15 ${CMAKE_CURRENT_SOURCE_DIR}/tests/microhydro_post_processing.arc)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
