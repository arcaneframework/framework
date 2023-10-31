# Comparaison des synchronisations {#arcanedoc_debug_perf_compare_synchronization}

Depuis la version 3.11 de %Arcane, il existe un mécanisme automatique
permettant de comparer les valeurs des variables avant et après une
synchronisation. Cela permet de savoir si une synchronisation est
utile ou pas.

\note Actuellement ce mécanisme ne fonctionne que pour les
synchronisations simples (celles qui sont appelées via la méthode
\arcane{MeshVariableRef::synchronize()}.

Pour activer ce mode, il faut positionner la variable d'environnement
`ARCANE_AUTO_COMPARE_SYNCHRONIZE`. Les trois valeurs possibles sont :

- `1` : pour activer le mécanisme et afficher en fin de calcul pour
  chaque variable le nombre de synchronisations qui ont modifiées les
  valeurs des mailles fantômes.
- `2` : comme `1` mais en plus il y a une impression listing au moment
  de la synchronisation si la synchronisation n'a pas modifié de
  valeurs (ce qui laisse supposer qu'elle n'est potentiellement pas
  utile).
- `3` : comme `2` mais en plus la pile d'appel au moment de la
  synchronisation est affichée.

A noter que les modes `2` et `3` nécessitent de faire une réduction pour
chaque synchronisation ce qui peut impacter les performances.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_compare_bittobit
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_unit_tests
</span>
</div>
