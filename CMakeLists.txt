cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# ----------------------------------------------------------------------------
# Indique que le projet n'a pas de language
# Cela est indispensable pour que CMake ne recherche pas les
# compilateurs avant d'avoir inclus les fichiers de configuration.
Project(ArcaneRepo LANGUAGES NONE)

# ----------------------------------------------------------------------------
# Le vrai projet se trouve dans un sous-répertoire.
# A terme, on déplacera le CMakeLists.txt de ce sous-répertoire ici
add_subdirectory(arcane)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
