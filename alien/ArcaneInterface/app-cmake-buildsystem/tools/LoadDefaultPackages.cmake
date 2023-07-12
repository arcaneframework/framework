
# mettre ici tous les packages chargés par défaut
# on cherche des fichiers Find**.cmake


if(USE_ARCCON)
  loadPackage(NAME Arccon ESSENTIAL)
  list(APPEND CMAKE_MODULE_PATH ${ARCCON_MODULE_PATH_M})
  if(USE_AXLSTAR)
    loadPackage(NAME Axlstar ESSENTIAL)

    if(NOT WIN32)
      loadPackage(NAME Mono ESSENTIAL)
    endif()

    loadPackage(NAME DotNet ESSENTIAL)
  endif()
  
  loadPackage(NAME Glib ESSENTIAL)
  if(TARGET arcconpkg_Glib)
    add_library(glib ALIAS arcconpkg_Glib)
  endif()

else()

  if(NOT WIN32)
    loadPackage(NAME Mono ESSENTIAL)
  endif()

  loadPackage(NAME DotNet ESSENTIAL)

  loadPackage(NAME GLib ESSENTIAL)
  loadPackage(NAME Glib ESSENTIAL)

endif()

include(${BUILD_SYSTEM_PATH}/commands/user/generateCMakeConfig.cmake)
