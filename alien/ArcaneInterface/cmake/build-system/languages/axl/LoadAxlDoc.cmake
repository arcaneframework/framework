if(${VERBOSE})
  logStatus("    ** Generate axldoc.exe")
endif()

add_custom_command(
  OUTPUT  ${OutputPath}/axldoc.exe
  COMMAND ${XBUILD} 
  ARGS    ${BUILD_SYSTEM_PATH}/csharp/axl/axldoc.csproj ${XBUILD_ARGS} 
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/axldoc.csproj ${OutputPath}/Arcane.Axl.dll
  )
   
bundle(
  BUNDLE ${PROJECT_BINARY_DIR}/bin/axldoc.exe 
  EXE    ${OutputPath}/axldoc.exe
  DLLs   ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  )

# génération de axldoc conditionnelle au début
add_custom_target(
  axldoc ALL DEPENDS axl
  ${PROJECT_BINARY_DIR}/bin/axldoc.exe
  ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  )

# on crée une target pour pouvoir écrire 
# /> make axldoc
add_custom_target(dotnet_axldoc
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.csproj ${XBUILD_ARGS}
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/axldoc.csproj    ${XBUILD_ARGS}
  COMMENT "generate axldoc tools")

install(FILES ${PROJECT_BINARY_DIR}/bin/axldoc.exe DESTINATION bin)
