
if(${VERBOSE})
  logStatus("    ** Generate Arcane.Axl.dll")
endif()

add_custom_command(
  OUTPUT  ${OutputPath}/Arcane.Axl.dll
  COMMAND ${XBUILD} 
  ARGS    ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.csproj ${XBUILD_ARGS} 
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.csproj
  )

add_custom_command(
  OUTPUT  ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  COMMAND ${CMAKE_COMMAND} -E 
  copy_if_different ${OutputPath}/Arcane.Axl.dll ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  DEPENDS ${OutputPath}/Arcane.Axl.dll
  )
  
# on crée une target pour pouvoir écrire 
# /> make dotnet_arcane_axl_dll
add_custom_target(dotnet_arcane_axl_dll
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.csproj ${XBUILD_ARGS}
  COMMENT "generate Arcane.Axl dll")
