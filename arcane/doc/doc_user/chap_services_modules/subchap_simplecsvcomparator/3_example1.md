# Exemple n°1 {#arcanedoc_services_modules_simplecsvcomparator_example1}

[TOC]

Dans ce premier exemple, on va réutiliser l'exemple 1 du sous-chapitre précédent
(\ref arcanedoc_services_modules_simplecsvoutput_example1) mais en modifiant
le point d'entrée `exit`.

On va utiliser les deux services en tant que singleton ici.

## Fichier .config

Pour commencer, voyons le `.config` :

`csv.config` `<time-loop name="example1">`
\snippet stc.config SimpleTableComparatorExample1_config

Ici, on voit les deux singletons que l'on va utiliser.

## Point d'entrée initial

Voici le point d'entrée `start-init` :

`SimpleTableComparatorExample1Module.cc`
\snippet SimpleTableComparatorExample1Module.cc SimpleTableComparatorExample1_init


## Point d'entrée loop

Le point d'entrée `compute-loop` :

`SimpleTableComparatorExample1Module.cc`
\snippet SimpleTableComparatorExample1Module.cc SimpleTableComparatorExample1_loop


## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableComparatorExample1Module.cc`
\snippet SimpleTableComparatorExample1Module.cc SimpleTableComparatorExample1_exit

On peut voir un exemple minimal d'utilisation du comparator.
On commence par récupérer le pointeur vers le singleton puis on initialise
le comparator en lui donnant un pointeur vers un objet ayant comme interface 
\arcane{ISimpleTableOutput}.

Ensuite, on regarde si un fichier de référence existe.
S'il n'y en a pas, on le crée avec les valeurs de `table`.  
En effet, le service `SimpleCsvComparator` est aussi capable d'écrire des 
fichiers. Il va utiliser les informations du \arcane{ISimpleTableOutput}
pour trouver le chemin.

Si on regarde dans le point d'entrée `init`, on peut voir l'initialisation
du service `SimpleCsvOutput` :

```cpp
table->init("Results_Example1", "example1");
```

`SimpleCsvComparator` va utiliser ces informations pour écrire le fichier
de référence. Dans cet exemple, il va aller écrire le fichier ici :
```sh
./output/csv_refs/example1/Results_Example1.csv
```

Si le fichier de référence existe déjà, le comparator va le comparer avec les
valeurs de l'objet `table`.  
Si les valeurs sont identiques, on aura le message `Mêmes valeurs !!!`
dans la sortie, sinon on aura `Valeurs différentes :(` (et un code d'erreur
`1`).

À la fin, le service `SimpleCsvOutput` écrit son fichier, comme d'habitude.

\remark
Vous l'aurez compris, il faut lancer l'exemple deux fois pour voir ce qu'il
se passe.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_examples
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_example2
</span>
</div>
