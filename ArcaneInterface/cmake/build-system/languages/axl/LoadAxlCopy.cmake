if(${VERBOSE})
  logStatus("    ** Generate axlcopy.exe")
endif()

add_custom_command(
  OUTPUT  ${OutputPath}/axlcopy.exe
  COMMAND ${XBUILD} 
  ARGS    ${BUILD_SYSTEM_PATH}/csharp/axl/axlcopy.csproj ${XBUILD_ARGS} 
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/axlcopy.csproj ${OutputPath}/Arcane.Axl.dll
  )
   
bundle(
  BUNDLE ${PROJECT_BINARY_DIR}/bin/axlcopy.exe 
  EXE    ${OutputPath}/axlcopy.exe
  DLLs   ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  )

# génération de axlcopy conditionnelle au début
add_custom_target(
  axlcopy ALL DEPENDS axl
  ${PROJECT_BINARY_DIR}/bin/axlcopy.exe
  ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  )

# on crée une target pour pouvoir écrire 
# /> make axlcopy
add_custom_target(dotnet_axlcopy
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.csproj ${XBUILD_ARGS}
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/axlcopy.csproj    ${XBUILD_ARGS}
  COMMENT "generate axlcopy tools")

install(FILES ${PROJECT_BINARY_DIR}/bin/axlcopy.exe DESTINATION bin)
