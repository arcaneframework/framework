# Exemple n°3 {#arcanedoc_services_modules_simplecsvcomparator_example3}

[TOC]

Avec l'exemple 3, on a mixé singleton et normal.  
Le service `SimpleCsvOutput` est en singleton, `SimpleCsvComparator` ne l'est pas.


## Point d'entrée initial

Point d'entrée `start-init` :

`SimpleTableComparatorExample3Module.cc`
\snippet SimpleTableComparatorExample3Module.cc SimpleTableComparatorExample3_init



## Point d'entrée loop

Point d'entrée `compute-loop` :

`SimpleTableComparatorExample3Module.cc`
\snippet SimpleTableComparatorExample3Module.cc SimpleTableComparatorExample3_loop


## Point d'entrée exit

Voyons le point d'entrée `exit` :

`SimpleTableComparatorExample3Module.cc`
\snippet SimpleTableComparatorExample3Module.cc SimpleTableComparatorExample3_exit

Ici, on a ajouté trois choses : deux `editElement()` et un `editRegexRows()`.  
Les `editElement()` permettent de modifier un élément pour avoir une erreur lors de la
comparaison.  
Le `editRegexRows()` permet de choisir les lignes que l'on veut comparer.

Dans ce cas, seule les lignes contenant `Fissions` dans leur nom sera comparé
(donc ici, uniquement la ligne `Nb de Fissions`).

En effet, on peut choisir quelles sont les lignes et les colonnes que l'on veut
comparer. On peut le faire via des expressions régulière mais aussi en spécifiant
directement leur nom (via les méthodes \arcane{ISimpleTableComparator::addRowForComparing()}
et \arcane{ISimpleTableComparator::addColumnForComparing()}).  
Il est aussi possible de spécifier que les lignes/colonnes que l'on donne sont
des lignes/colonnes que l'on veut exclure de la comparaison (méthodes
\arcane{ISimpleTableComparator::isAnArrayExclusiveRows()},
\arcane{ISimpleTableComparator::isAnArrayExclusiveColumns()},
\arcane{ISimpleTableComparator::isARegexExclusiveRows()} et
\arcane{ISimpleTableComparator::isARegexExclusiveColumns()}).

Enfin, il est possible de spécifier un épsilon pour avoir une marge d'erreur acceptable
(\arcane{ISimpleTableComparator::addEpsilonRow()} /
\arcane{ISimpleTableComparator::addEpsilonColumn()}).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example2
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example4
</span> -->
</div>
