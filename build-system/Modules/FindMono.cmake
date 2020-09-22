#
# Find the 'mono' executable
#
# This module defines
# MONO_EXEC , the path of the 'mono' executable
# Mono_FOUND, if 'mono' is found

# TODO: n'appeler qu'une seule fois ce package.

if (NOT Mono_FOUND)
  find_program(MONO_EXEC mono)
  logStatus("Mono: MONO_EXEC is '${MONO_EXEC}'")
  if (MONO_EXEC)
    # Recherche le numéro de version de Mono.
    # Pour cela, lance la commande 'mono --version' et
    # analyse la chaîne retournée qui est du style:
    # Mono JIT compiler version 4.0.4 (Stable 4.0.4.1/5ab4c0d Thu Jun  1 08:07:07 CEST 2017)
    execute_process(COMMAND ${MONO_EXEC} "--version" OUTPUT_VARIABLE MONO_EXEC_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "compiler version ([0-9]+)\.([0-9]+)" MONO_VERSION_REGEX_MATCH ${MONO_EXEC_VERSION_OUTPUT})
    set(MONO_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
    logStatus("Mono: MONO_VERSION = ${MONO_VERSION}")

    set(MONO_FOUND TRUE)
    set(Mono_FOUND TRUE)
  endif()
endif()

find_program(ALIEN_MONO_MKBUNDLE mkbundle)

set(ALIEN_MONO_MKBUNDLE_OPTIONS)
set(Mkbundle_EXEC)
if (ALIEN_MONO_MKBUNDLE)
  set(Mkbundle_EXEC ${CMAKE_CURRENT_BINARY_DIR}/mkbundle.exe)
	#check if we can specify i18n
	execute_process(COMMAND ${ALIEN_MONO_MKBUNDLE} --help
					OUTPUT_VARIABLE MKBUNDLE_HELP)
	if(${MKBUNDLE_HELP} MATCHES "--i18n")
		SET(ALIEN_MONO_MKBUNDLE_OPTIONS "${ALIEN_MONO_MKBUNDLE_OPTIONS} --i18n none")
	endif()
endif (ALIEN_MONO_MKBUNDLE)

include(FindPackageHandleStandardArgs)

# pour limiter le mode verbose
set(Mono_FIND_QUIETLY ON)

find_package_handle_standard_args(Mono
  DEFAULT_MSG 
  MONO_EXEC 
  Mkbundle_EXEC
  )

get_filename_component(MONO_EXEC_PATH ${MONO_EXEC} PATH)

find_file(ALIEN_MONO_PKG_CONFIG_PATH
	  NAMES pkgconfig
	  HINTS ${MONO_EXEC_PATH}/../lib
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/mkbundle.exe.in mkbundle.exe @ONLY)

get_filename_component(MONOembed_ROOT_PATH ${MONO_EXEC} PATH)
 
mark_as_advanced(MONOembed_ROOT_PATH)
