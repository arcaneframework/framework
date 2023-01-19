# Détection des problèmes mémoire {#arcanedoc_debug_perf_check_memory}

Arcane dispose d'un mécanisme permettant de détecter certains problèmes
mémoires, en particulier :
- les fuites mémoires
- les désallocations qui ne correspondent à aucune allocation.

De plus, cela permet d'obtenir des statistiques sur l'utilisation
mémoire.

\warning Ce mécanisme ne fonctionne actuellement que sur les OS Linux.

\warning Ce mécanisme ne fonctionne pas lorsque le multi-treading est activé.

Pour l'activer, il suffit de positionner la variable d'environnement
ARCANE_CHECK_MEMORY à \c true. Toutes les allocations et désallocations
sont tracées. Cependant, pour des problèmes de performance, on ne
conserve et n'affiche la pile d'appel que pour les allocations supérieures
à une certaine taille. Par défaut, la valeur est de 1Mo (1000000). Il est possible
de spécifier une autre valeur via la variable d'environnement
ARCANE_CHECK_MEMORY_BLOCK_SIZE. La variable d'environnement
ARCANE_CHECK_MEMORY_BLOCK_SIZE_ITERATION permet de spécifier une valeur
de bloc qui sera utilisé pour la boucle en temps après
l'initialisation. Cela permet de tracer plus finement les allocations
durant le calcul que celles qui ont lieu lors de l'initialisation.

Les appels sont tracés depuis l'appel à ArcaneMain::arcaneInitialize()
jusqu'à l'appel à ArcaneMain::arcaneFinalize(). Lors de ce dernier appel,
une liste des blocs alloués qui n'ont pas été désalloués est affiché.

Il est possible de gérer plus finement le vérificateur mémoire
via l'interface IMemoryInfo. Cette interface est un singleton accessible
via la méthode arcaneGlobalMemoryInfo();

\note INTERNE: Pour l'instant, les éventuelles incohérences entre allocation
et désallocations sont indiquées sur std::cout. Cela peut poser des problèmes
de lisibilités en parallèle. \`A terme, il faudra utiliser ITraceMng, mais
cela est délicat actuellement car ce mécanisme effectue lui aussi des
appels mémoire et il est difficile de le rendre compatible avec les fonctions
de débug actuelles.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_compare_bittobit
</span>
</div>
