# La méthode 'bundle' est déclarée dans 'ArcconDotNet' et s'appelle 'arccon_dotnet_mkbundle'.
message(FATAL_ERROR "Do not include. Use 'ArcconDotNet.cmake" instead)

# function(bundle)

#   set(options)
#   set(oneValueArgs BUNDLE EXE)
#   set(multiValueArgs DLLs)
 
#   cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
#   if(ARGS_UNPARSED_ARGUMENTS)
#     logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
#   endif()
  
#   get_filename_component(File ${ARGS_BUNDLE} NAME) 

#   if(WIN32)
#     add_custom_command(
#       OUTPUT  ${ARGS_BUNDLE} 
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ARGS_EXE} ${ARGS_BUNDLE}
#       DEPENDS ${ARGS_EXE} ${ARGS_DLLs}
#       )
#   else()
#     # mkbundle ne peut être lancé en parallèle dans un même répertoire
#     # on utilise un répertoire temporaire et WORKING_DIRECTORY de add_custom_command
#     # L'option '--skip-scan' permet à mkbundle de ne pas lever d'exceptions si une des assembly
#     # n'est pas lisible.
#     file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/dotnet/tmp/${File})
#     add_custom_command(
#       OUTPUT  ${ARGS_BUNDLE} 
#       COMMAND ${Mkbundle_EXEC} ${ARGS_EXE} ${ARGS_DLLs} -o ${ARGS_BUNDLE} --deps --static --skip-scan
#       DEPENDS ${ARGS_EXE} ${ARGS_DLLs} ${Mkbundle_EXEC}
#       WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/dotnet/tmp/${File}
#       )
#   endif()
  
# endfunction()
