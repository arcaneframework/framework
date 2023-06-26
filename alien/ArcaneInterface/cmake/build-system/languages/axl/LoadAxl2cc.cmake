if(${VERBOSE})
  logStatus("    ** Generate axl2cc.exe")
endif()
	
add_custom_command(
  OUTPUT  ${OutputPath}/axl2cc.exe
  COMMAND ${XBUILD} 
  ARGS    ${BUILD_SYSTEM_PATH}/csharp/axl/axl2cc.csproj ${XBUILD_ARGS} 
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/axl2cc.csproj ${OutputPath}/Arcane.Axl.dll
  )
   
bundle(
  BUNDLE ${PROJECT_BINARY_DIR}/bin/axl2cc.exe 
  EXE    ${OutputPath}/axl2cc.exe
  DLLs   ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  )

add_custom_command(
  OUTPUT  ${PROJECT_BINARY_DIR}/share/axl.xsd
  COMMAND ${CMAKE_COMMAND} -E 
  copy_if_different ${BUILD_SYSTEM_PATH}/csharp/axl/axl.xsd ${PROJECT_BINARY_DIR}/share/axl.xsd
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/axl.xsd
  )

# génération de axl2cc conditionnelle au début
add_custom_target(
  axl ALL DEPENDS 
  ${PROJECT_BINARY_DIR}/bin/axl2cc.exe
  ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  ${PROJECT_BINARY_DIR}/share/axl.xsd
  )

# on crée une target pour pouvoir écrire 
# /> make axl2cc
add_custom_target(dotnet_axl2cc 
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.csproj ${XBUILD_ARGS}
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/axl2cc.csproj     ${XBUILD_ARGS}
  COMMENT "generate axl2cc tools")

install(FILES ${PROJECT_BINARY_DIR}/bin/axl2cc.exe DESTINATION bin)

# répertoire de sortie des axl
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/share/axl)

# installation de la xsd des fichiers axl
install(FILES ${PROJECT_BINARY_DIR}/share/axl.xsd DESTINATION share)

set(AXL2CC ${PROJECT_BINARY_DIR}/bin/axl2cc.exe)
set(AXL2CCT4 FALSE)