find_package(LibXml2 REQUIRED)

include(srcs.cmake)

arcane_add_library(arcane_core
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/core
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_core PRIVATE ARCANE_COMPONENT_arcane_core)

target_link_libraries(arcane_core PUBLIC arcane_accelerator_core arcane_utils)
target_link_libraries(arcane_core PUBLIC Arccore::arccore_message_passing Arccore::arccore_serialize)
target_link_libraries(arcane_core PRIVATE LibXml2::LibXml2)
target_include_directories(arcane_core PUBLIC $<BUILD_INTERFACE:${COMPAT_DIRECTORY}>)

arcane_register_library(arcane_core)

# Génère pour chaque include 'Toto.h' un fichier d'en-tête
# 'arcane/Toto.h' qui inclue 'arcane/core/Toto.h'
# Cela est nécessaire pour les codes utilisant les versions
# d'Arcane pour lesquelles les fichiers d'en-tête n'étaient pas
# dans 'core'.
set(SED_REGEX)
foreach(file ${ARCANE_ORIGINAL_SOURCES})
  if (${file} MATCHES "\.(h|H)$")
    string(TOUPPER ${file} upper_name1)
    string(REPLACE "." "_" upper_name2 ${upper_name1})
    string(REPLACE "/" "_" upper_name ${upper_name2})
    get_filename_component(_file_dir "${file}" DIRECTORY [CACHE])
    configure_file(${Arcane_SOURCE_DIR}/cmake/IncludeCompat.h.in "${ARCANE_COMPATIBILITY_DIRECTORY}/arcane/${file}" @ONLY)
    install(FILES "${ARCANE_COMPATIBILITY_DIRECTORY}/arcane/${file}" DESTINATION "include/arcane/${_file_dir}")
    set(SED_REGEX "${SED_REGEX}s-#include \\\"arcane/${file}\\\"-#include \\\"arcane/core/${file}\\\"-g\n")
  endif()
endforeach()
# Génère un fichier contenant des expressions régulières pour modifier
# un fichier en remplacant les 'arcane/Toto.h' par 'arcane/core/Toto.h'
file(WRITE ${CMAKE_BINARY_DIR}/share/change_core_header_sed_regex.txt ${SED_REGEX})
file(WRITE ${CMAKE_BINARY_DIR}/share/do_change_core_header_sed_regex.sh "#!/bin/sh\nsed -i --file=\"${CMAKE_BINARY_DIR}/share/change_core_header_sed_regex.txt\" \"$@\"\n")

if (GTEST_FOUND)
  add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
