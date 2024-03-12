# ----------------------------------------------------------------------------
# Liste de noms potentiels de cibles pour des packages
#
# Pour les packages de cette liste, l'appel à arcane_find_package() indiquera
# NOT FOUND si aucune cible de cette liste n'est trouvé.

list(APPEND ARCANE_PACKAGE_SEARCH_TARGETS_ML "trilinos_ml" "ml")
list(APPEND ARCANE_PACKAGE_SEARCH_TARGETS_AztecOO "trilinos_aztecoo" "aztecoo")
list(APPEND ARCANE_PACKAGE_SEARCH_TARGETS_Epetra "trilinos_epetra" "epetra")
list(APPEND ARCANE_PACKAGE_SEARCH_TARGETS_Ifpack "trilinos_ifpack" "ifpack")

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
