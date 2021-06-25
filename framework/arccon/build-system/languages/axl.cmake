# NOTE: Ce fichier n'est plus utilisé mais on le garde temporairement pour
# ensuite l'include en partie dans 'axlstar' (pour ce qui concerne l'utilisation
# de 'mkbundle').

include(${BUILD_SYSTEM_PATH}/ArcconDotNet.cmake)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(${VERBOSE})
  logStatus("    ** Generate axl2ccT4.exe")
endif()
	
add_custom_command(
  OUTPUT  ${OutputPath}/axl2ccT4.exe
  COMMAND ${XBUILD} 
  ARGS    ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.T4/axl2ccT4.csproj ${XBUILD_ARGS} 
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.T4/axl2ccT4.csproj ${OutputPath}/Arcane.Axl.dll
  )
   
arccon_dotnet_mkbundle(
  BUNDLE ${PROJECT_BINARY_DIR}/bin/axl2ccT4.exe 
  EXE    ${OutputPath}/axl2ccT4.exe
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
  ${PROJECT_BINARY_DIR}/bin/axl2ccT4.exe
  ${PROJECT_BINARY_DIR}/bin/Arcane.Axl.dll
  ${PROJECT_BINARY_DIR}/share/axl.xsd
  )

# on crée une target pour pouvoir écrire 
# /> make axl2cc
add_custom_target(dotnet_axl2cc 
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.csproj ${XBUILD_ARGS}
  COMMAND ${XBUILD} ${BUILD_SYSTEM_PATH}/csharp/axl/Arcane.Axl.T4/axl2ccT4.csproj     ${XBUILD_ARGS}
  COMMENT "generate axl2cc tools")

install(FILES ${PROJECT_BINARY_DIR}/bin/axl2ccT4.exe DESTINATION bin)

# répertoire de sortie des axl
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/share/axl)

# installation de la xsd des fichiers axl
install(FILES ${PROJECT_BINARY_DIR}/share/axl.xsd DESTINATION share)

set(AXL2CC ${PROJECT_BINARY_DIR}/bin/axl2ccT4.exe)
set(AXL2CCT4 TRUE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(${VERBOSE})
  logStatus("    ** Generate axlcopy.exe")
endif()

add_custom_command(
  OUTPUT  ${OutputPath}/axlcopy.exe
  COMMAND ${XBUILD} 
  ARGS    ${BUILD_SYSTEM_PATH}/csharp/axl/axlcopy.csproj ${XBUILD_ARGS} 
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/axlcopy.csproj ${OutputPath}/Arcane.Axl.dll
  )
   
arccon_dotnet_mkbundle(
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

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(${VERBOSE})
  logStatus("    ** Generate axldoc.exe")
endif()

add_custom_command(
  OUTPUT  ${OutputPath}/axldoc.exe
  COMMAND ${XBUILD} 
  ARGS    ${BUILD_SYSTEM_PATH}/csharp/axl/axldoc.csproj ${XBUILD_ARGS} 
  DEPENDS ${BUILD_SYSTEM_PATH}/csharp/axl/axldoc.csproj ${OutputPath}/Arcane.Axl.dll
  )
   
arccon_dotnet_mkbundle(
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
