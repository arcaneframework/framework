# Exemple n°5 {#arcanedoc_services_modules_simplecsvoutput_example5}

[TOC]

Dans cet exemple, on va montrer un exemple d'utilisation de lecture/
remplissage du tableau via la position d'un pointeur.

En effet, jusque-là, on a rempli nos tableaux de valeurs avec les
noms de lignes/colonnes, en mettant les valeurs les une à côté des
autres.  
Il se trouve qu'en interne, lorsqu'un élément est ajouté ou modifié,
il y a un pointeur qui se met à jour et qui pointe vers le dernier
élément manipulé.

\remark
Pointeur dans le sens "pointeur vers une position du tableau 2D",
pas dans le sens "pointeur `C` vers une zone mémoire".

On va donc pouvoir utiliser ce pointeur pour modifier des valeurs
autour de la dernière valeur manipulée.

Results_Example5             |Iteration 1|Iteration 2|Iteration 3|Somme
-----------------------------|-----------|-----------|-----------|-----------
Nb de Fissions               |36         |0          |85         |121
Nb de Fissions (div par 2)   |18         |0          |42.5       |60.5
Nb de Collisions             |29         |84         |21         |134
Nb de Collisions (div par 2) |14.5       |42         |10.5       |67



## Point d'entrée initial

Voyons le point d'entrée `start-init` :

`SimpleTableOutputExample5Module.cc`
\snippet SimpleTableOutputExample5Module.cc SimpleTableOutputExample5_init

Ici, rien de bien original, mis à part les deux lignes en plus : `(div par 2)`.
Ces deux lignes vont contenir le nombre de fissions/collisions d'une itération
divisé par 2.



## Point d'entrée loop

Voyons le point d'entrée `compute-loop` :

`SimpleTableOutputExample5Module.cc`
\snippet SimpleTableOutputExample5Module.cc SimpleTableOutputExample5_loop

Une nouvelle méthode fait son apparition : \arcane{ISimpleTableOutput::editElementDown()}.  
Cette méthode permet de modifier la "case" sous la "case" que l'on vient de
modifier.  
Prenons les deux lignes dédiées à la fission. La première ligne ajoute une
valeur sur la ligne `Nb de Fissions`.  
En interne, un pointeur est modifié et pointe désormais sur la valeur que
l'on vient d'ajouter.  
La seconde ligne appelle la nouvelle méthode. Cette méthode va prendre le
pointeur, va rechercher la "case" en dessous et va remplacer sa valeur par
`nb_fissions/2`. Par défaut, le pointeur sera alors mis à jour et pointera
vers cette "case" manipulée. Donc si on voulait ajouter, par exemple, `nb_fissions*2`
juste en dessous, on pourrai refaire un appel à la méthode
\arcane{ISimpleTableOutput::editElementDown()} juste après.

Dans le cas où il y aurai une modification de `nb_fissions` entre les
deux lignes (sans toucher au tableau) :

```cpp
options()->csvOutput()->addElementInRow(pos_fis, nb_fissions);
nb_fissions += 456;
options()->csvOutput()->editElementDown(nb_fissions/2.); // Pas la bonne valeur !!!
```
On pourrai faire ça :

```cpp
options()->csvOutput()->addElementInRow(pos_fis, nb_fissions);
nb_fissions += 456;
options()->csvOutput()->editElementDown(element()/2.); // C'est correct !!!
```

\arcane{ISimpleTableOutput::element()} est une méthode qui permet de récupérer
la valeur de la "case" pointé par le pointeur. Ça peut être pratique dans ce
cas là par exemple (on est d'accord que si on n'utilise pas le pointeur et
les méthodes associées, c'est une méthode qui ne sert à rien).

\note
Il n'y a pas que la méthode \arcane{ISimpleTableOutput::editElementDown()},
il existe des méthodes équivalentes pour les quatre directions :  
\arcane{ISimpleTableOutput::editElementUp()}  
\arcane{ISimpleTableOutput::editElementLeft()}  
\arcane{ISimpleTableOutput::editElementRight()}  
Idem pour \arcane{ISimpleTableOutput::element()}.


## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableOutputExample5Module.cc`
\snippet SimpleTableOutputExample5Module.cc SimpleTableOutputExample5_exit

Ce point d'entrée est identique à celui de l'exemple précédent.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example4
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example6
</span>
</div>
