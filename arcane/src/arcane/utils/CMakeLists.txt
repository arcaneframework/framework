# TODO: a supprimer lorsqu'on n'utilisera plus 'arcane.pc'.
macro(arcane_set_external_includes)
 SET(_PKGCONFIG_INCLUDE_DIRS "")
 foreach(_my_var ${ARGN})
  message(STATUS "Including extra path for next component parsing: ${_my_var}")
  SET(_PKGCONFIG_INCLUDE_DIRS "${_PKGCONFIG_INCLUDE_DIRS} -I${_my_var}")
 endforeach(_my_var)
 # MESSAGE(STATUS "MY_INC ${_PKGCONFIG_INCLUDE_DIRS}")
endmacro(arcane_set_external_includes)

arcane_set_external_includes(${ARCANE_SRC_PATH} ${GLIB_INCLUDE_DIRS})

include(srcs.cmake)

arcane_add_library(arcane_utils
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/utils
  FILES ${ARCANE_SOURCES}
  )

arcane_add_arccon_packages(arcane_utils PRIVATE ${PKGS})

target_compile_definitions(arcane_utils PRIVATE ARCANE_COMPONENT_arcane_utils)

target_include_directories(arcane_utils PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)
target_include_directories(arcane_utils PUBLIC $<BUILD_INTERFACE:${ARCANE_SRC_PATH}> $<INSTALL_INTERFACE:include>)

# 'arcane_build_compile_flags' n'est utilisé que lors de la compilation
# de la bibliothèque mais ne doit pas être exportée.
target_link_libraries(arcane_utils PUBLIC $<BUILD_INTERFACE:arcane_build_compile_flags> arcane_export_compile_flags)

# Ajoute dépendances provenant de 'Arccore'
target_link_libraries(arcane_utils PUBLIC Arccore::arccore_trace Arccore::arccore_concurrency Arccore::arccore_collections Arccore::arccore_base arccon::Glib)

# ----------------------------------------------------------------------------
# Liste des fonctionnalités du C++11 utilisées.
# Cette liste est un sous-ensemble de la variable CMAKE_CXX_KNOWN_FEATURES
# GG NOTE: Actuellement (version 3.2), on utilise les fonctionnalités suivantes
# du C++11: cxx_inline_namespaces, cxx_deleted_functions.
# Cependant, avec cmake 3.3.2, ces features sont considérées comme
# non supportées par Visual Studio 2013 alors que ce n'est pas le cas
# (avec VS2013 Update 4, cela fonctionne). On les enlève donc provisoirement
# de la liste. A noter qu'avec VS2015, ces fonctionnalités sont correctement
# reconnuees.
target_compile_features(arcane_utils PUBLIC cxx_std_14 cxx_lambdas cxx_auto_type cxx_nullptr cxx_rvalue_references)

arcane_register_library(arcane_utils)

message(STATUS "END OF PARSE UTILS")
message(STATUS $<TARGET_PROPERTY:arcane_utils,INTERFACE_INCLUDE_DIRECTORIES>)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
