# Package 'DbgHelp' de windows qui permet d'accéder aux informations
# de débug (notamment pile d'appel)

# TODO: ne faire que sur x64 ou détecter l'architecture

set(DbgHelp_FOUND FALSE)

if (NOT WIN32)
  return()
endif()
if (NOT DEFINED CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
  return()
endif()

set(_WINDOWS_SDK_BASE_PATH "C:/Program Files (x86)/Windows Kits/10")
set(_WINDOWS_SDK_LIB_PATH "${_WINDOWS_SDK_BASE_PATH}/Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um/x64")
set(_WINDOWS_SDK_INCLUDE_PATH "${_WINDOWS_SDK_BASE_PATH}/Include/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um")

find_library(DbgHelp_LIBRARY DbgHelp HINTS ${_WINDOWS_SDK_LIB_PATH})
find_path(DbgHelp_INCLUDE_DIR DbgHelp.h HINTS ${_WINDOWS_SDK_INCLUDE_PATH})

message(STATUS "DbgHelp_LIBRARY=${DbgHelp_LIBRARY}")
message(STATUS "DbgHelp_INCLUDE_DIR=${DbgHelp_INCLUDE_DIR}")

if (DbgHelp_INCLUDE_DIR AND DbgHelp_LIBRARY)
  set(DbgHelp_FOUND TRUE)
  set(DbgHelp_LIBRARIES ${DbgHelp_LIBRARY} )
  set(DbgHelp_INCLUDE_DIRS ${DbgHelp_INCLUDE_DIR})
endif()

arccon_register_package_library(DbgHelp DbgHelp)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
