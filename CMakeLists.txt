cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# ----------------------------------------------------------------------------
# Indique que le projet n'a pas de language
# Cela est indispensable pour que CMake ne recherche pas les
# compilateurs avant d'avoir inclus les fichiers de configuration.
Project(ArcaneRepo LANGUAGES NONE)

# ----------------------------------------------------------------------------

enable_testing()

# Obsolète: permet de choisir quel composant on compile
# Il faut maintenant utiliser ARCANEFRAMEWORK_BUILD_COMPONENTS
# qui est utilisé dans _common/build_all/CMakeLists.txt
if (NOT FRAMEWORK_BUILD_COMPONENT)
  set (FRAMEWORK_BUILD_COMPONENT all)
endif()

option(FRAMEWORK_NO_EXPORT_PACKAGES "True if we don't want to export packages found by arccon in {Arccore|Arcane}Targets.cmake." ON)

if (FRAMEWORK_BUILD_COMPONENT STREQUAL all)
  # Compile Arcane et ses composantes en même temps
  add_subdirectory(_common/build_all)
elseif (FRAMEWORK_BUILD_COMPONENT STREQUAL arcane)
  # Compile uniquement Arcane.
  add_subdirectory(arcane)
elseif (FRAMEWORK_BUILD_COMPONENT STREQUAL arccon)
  add_subdirectory(arccon)
elseif (FRAMEWORK_BUILD_COMPONENT STREQUAL arccore)
  add_subdirectory(arccore)
elseif (FRAMEWORK_BUILD_COMPONENT STREQUAL axlstar)
  add_subdirectory(axlstar)
elseif (FRAMEWORK_BUILD_COMPONENT STREQUAL alien)
  add_subdirectory(alien)
elseif (FRAMEWORK_BUILD_COMPONENT STREQUAL alien_standalone)
  set(ALIEN_BUILD_COMPONENT alien_standalone)
  add_subdirectory(alien)
elseif (FRAMEWORK_BUILD_COMPONENT STREQUAL neo)
  add_subdirectory(arctools/neo)
else()
  message(FATAL_ERROR "Invalid value '${FRAMEWORK_BUILD_COMPONENT}' for FRAMEWORK_BUILD_COMPONENT"
    "Valid values are: { 'all', 'arcane', 'arccore', 'arccon', 'axlstar', 'alien', 'alien_standalone','neo' }")
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
