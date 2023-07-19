# -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
# ATTENTION: Pour compiler les exemples depuis le répertoire de compilation,
# il faut avoir auparavant installer 'Arcane'.

# Copie le répertoire 'samples' à l'identique dans le répertoire
# de compilation et d'installation
file(COPY samples DESTINATION ${CMAKE_BINARY_DIR})

# Il est important lors de la copie de mettre le setuid pour le groupe
# sinon il est possible suivant les OS et les répertoires d'installation
# que les fichiers créés sous les répertoires n'aient pas le bon groupe
# (ce qui empêche l'installation si des quotas sont prévus par groupe)
install(DIRECTORY samples
  DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE SETGID
  DESTINATION ${CMAKE_INSTALL_PREFIX})

# ----------------------------------------------------------------------------
# Génère un fichier de configuration contenant les noms des compilateurs
# et options de compilation utilisés par Arcane. Cela permet de garantir
# que les exemples utiliseront les mêmes versions de compilateur
# que ceux utilisés pour compiler Arcane.

# Configuration pour tester les exemples.
set(_CONFIG_ARCANE_ROOT "${CMAKE_INSTALL_PREFIX}/cmake")
configure_file(ArcaneCompilerConfig.cmake.in ${CMAKE_BINARY_DIR}/samples/ArcaneCompilerConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/samples/ArcaneCompilerConfig.cmake DESTINATION samples)

configure_file(nuget.config.in ${CMAKE_BINARY_DIR}/samples/nuget.config @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/samples/nuget.config DESTINATION samples)

# ----------------------------------------------------------------------------
# Génère des cibles pour configurer, compiler et tester les exemples
# NOTE: Il est nécessaire d'avoir installer Arcane pour utiliser ces cibles

set(ARCANE_SAMPLES_BUILD_DIR ${CMAKE_BINARY_DIR}/builddir_samples)
file(MAKE_DIRECTORY ${ARCANE_SAMPLES_BUILD_DIR})

# Cible pour configurer les exemples depuis le répertoire des sources
# (Cela est utile pour ne pas avoir à reconfigurer à chaque fois qu'on
# fait une modifications dans les exemples)
add_custom_target(samples_configure_insource
  COMMAND ${CMAKE_COMMAND} -C${CMAKE_BINARY_DIR}/samples/ArcaneCompilerConfig.cmake -DBUILD_SHARED_LIBS=TRUE ${CMAKE_CURRENT_SOURCE_DIR}/samples
  WORKING_DIRECTORY ${ARCANE_SAMPLES_BUILD_DIR}
  )

# Cible pour configurer les exemples depuis le répertoire de compilation
add_custom_target(samples_configure
  COMMAND ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR}/samples
  WORKING_DIRECTORY ${ARCANE_SAMPLES_BUILD_DIR}
  )

# Cible pour compiler les exemples.
# Il faut avoir lancer la cible 'samples_configure_insource'
# ou 'samples_configure' avant
add_custom_target(samples_build
  COMMAND ${CMAKE_COMMAND} --build . -v --clean-first
  WORKING_DIRECTORY ${ARCANE_SAMPLES_BUILD_DIR}
  )

# Cible pour lancer les tests des exemples.
add_custom_target(samples_tests
  COMMAND ${CMAKE_CTEST_COMMAND}
  WORKING_DIRECTORY ${ARCANE_SAMPLES_BUILD_DIR}
  )
