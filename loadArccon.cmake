include(${CMAKE_CURRENT_LIST_DIR}/Arccon.cmake)

include(LoadBuildSystem)

# options par défaut (verbose, cxx11)
include(LoadDefaultOptions)

# fichier des roots de packages (si défini)
include(LoadDefaultPackageFile)

# metas par défaut (win32/linux)
include(LoadDefaultMetas)

# options de compilation par défaut
include(LoadDefaultCompilationFlags)

# packages par défaut (mono et glib)
include(LoadDefaultPackages)

# langages par défaut (axl) NB: après les packages
# include(LoadDefaultLanguages)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
