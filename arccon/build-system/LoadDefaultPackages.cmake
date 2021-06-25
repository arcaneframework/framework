
# mettre ici tous les packages chargés par défaut
# on cherche des fichiers Find**.cmake

if(NOT WIN32)
  loadPackage(NAME Mono ESSENTIAL)
endif()

loadPackage(NAME DotNet ESSENTIAL)
loadPackage(NAME GLib ESSENTIAL)
