if (DotNetCore_FOUND)
  return()
endif()

find_program(DOTNET_EXEC NAMES dotnet PATHS)
message(STATUS "DOTNET exe: ${DOTNET_EXEC}")
if (NOT DOTNET_EXEC)
  message(FATAL_ERROR "no 'dotnet' exec found")
endif()

#TODO: rechercher la version

set(DotNetCore_FOUND TRUE)
