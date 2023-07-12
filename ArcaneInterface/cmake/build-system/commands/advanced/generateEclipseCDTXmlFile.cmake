function(generateEclipseCDTXmlFile)
 
  execute_process(COMMAND ${MONO_EXEC} 
    ${ECLIPSECDT_GENERATOR}
    ${PROJECT_BINARY_DIR}/pkglist.xml
    ${PROJECT_BINARY_DIR}/eclipse-config.xml
    )

endfunction()
