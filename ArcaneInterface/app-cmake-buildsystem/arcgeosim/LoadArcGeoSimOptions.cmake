# options
# --------

# Chargement des options de ArcGeoSim

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# options par défaut
include(LoadDefaultOptions)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(NOT ARCANE_VERSION)
  set(ARCANE_VERSION 2.2.1)
endif()

if(NOT ALIEN_VERSION)
  set(ALIEN_VERSION 1.1)
endif()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Version de arcane
createStringOption(COMMANDLINE ArcaneVersion
                   NAME        CUSTOM_ARCANE_VERSION
                   MESSAGE     "Arcane version"
                   DEFAULT     ${ARCANE_VERSION})

set(ARCANE_VERSION ${CUSTOM_ARCANE_VERSION_VALUE})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Version de arcane
createStringOption(COMMANDLINE AlienVersion
                   NAME        CUSTOM_ALIEN_VERSION
                   MESSAGE     "Alien version"
                   DEFAULT     ${ALIEN_VERSION})

set(ALIEN_VERSION ${CUSTOM_ALIEN_VERSION_VALUE})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Chemin de arcane
createStringOption(COMMANDLINE ArcanePath
                   NAME        CUSTOM_ARCANE_PATH
                   MESSAGE     "Arcane path")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Chemin de arcane
createStringOption(COMMANDLINE AlienPath
                   NAME        CUSTOM_ALIEN_PATH
                   MESSAGE     "Alien path")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
