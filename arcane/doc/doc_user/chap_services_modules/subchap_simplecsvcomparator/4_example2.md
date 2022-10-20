# Exemple n°2 {#arcanedoc_services_modules_simplecsvcomparator_example2}

[TOC]

Cet exemple 2 est identique à l'exemple 1, sans l'utilisation de singletons.


## Fichier .axl -- Partie <options>

Pour commencer, voici les options du fichier axl :

`SimpleTableComparatorExample2.axl`
\snippet SimpleTableComparatorExample2.axl SimpleTableComparatorExample2_options

On peut constater la présence des deux services.


## Fichier .arc -- Partie option du module

Voici le `.arc` correspondant :

`SimpleTableComparatorExample2.arc`
\snippet SimpleTableComparatorExample2.arc SimpleTableComparatorExample2_arc

Ce que l'on peut voir ici est que le comparator n'a pas d'option.


## Point d'entrée initial

Voici le point d'entrée `start-init` :

`SimpleTableComparatorExample2Module.cc`
\snippet SimpleTableComparatorExample2Module.cc SimpleTableComparatorExample2_init



## Point d'entrée loop

Le point d'entrée `compute-loop` :

`SimpleTableComparatorExample2Module.cc`
\snippet SimpleTableComparatorExample2Module.cc SimpleTableComparatorExample2_loop



## Point d'entrée exit

Enfin, voici le point d'entrée `exit` :

`SimpleTableComparatorExample2Module.cc`
\snippet SimpleTableComparatorExample2Module.cc SimpleTableComparatorExample2_exit

Pas de surprises, c'est la même utilisation.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example1
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example3
</span>
</div>
