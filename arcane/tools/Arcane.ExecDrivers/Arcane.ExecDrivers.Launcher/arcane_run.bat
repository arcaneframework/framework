rem set PATH=@CMAKE_BINARY_DIR@/lib:%PATH%
call "@DOTNET_EXEC@" "@_ARCANE_DOTNET_PUBLISH_DIR@/Arcane.ExecDrivers.Launcher.dll" %*
