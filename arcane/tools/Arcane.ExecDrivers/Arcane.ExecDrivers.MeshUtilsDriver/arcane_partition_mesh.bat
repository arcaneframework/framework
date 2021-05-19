rem LD_LIBRARY_PATH=@CMAKE_BINARY_DIR@/lib:${LD_LIBRARY_PATH}
call "@DOTNET_EXEC@" "@_ARCANE_DOTNET_PUBLISH_DIR@/Arcane.UnifiedDriver.dll" mesh_utils partition %*
