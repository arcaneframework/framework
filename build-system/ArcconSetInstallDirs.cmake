# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Force par défaut l'installation dans 'lib'.
# Idéalement il ne faudrait pas le faire mais avec le cmake actuel (3.16),
# l'inclusion de 'GNUInstallDirs' ne fonctionne pas toujours si le projet
# n'a pas de langage. Ceci est du au fait que 'GNUInstallDirs' essaie de
# détecter la plateforme (32 ou 64 bits) et que cela n'est pas possible
# sans compilateur. Bien entendu il y a une exception qui est 'debian' où
# il peut détecter cela sans compilateur (mais après normalement il faut
# spécifier l'architecture sous une autre forme).
#
# Bref, pour éviter tout problème (et aussi garder la compatibilité Windows),
# on positionne en dur 'lib'.
#
set(CMAKE_INSTALL_LIBDIR "lib")
include(GNUInstallDirs)
message(STATUS "INSTALL_LIBDIR = ${CMAKE_INSTALL_LIBDIR}")

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
