# Exemple n°4 {#arcanedoc_services_modules_simplecsvoutput_example4}

[TOC]

À partir de cet exemple, il ne sera plus question d'options ou de
singleton.
Ici, il sera question d'optimisation, dans le cas où ce service est
utilisé de manière plus sérieuse que du simple debuggage.

Le résultat va donc changer un peu :
Results_Example4|Iteration 1|Iteration 2|Iteration 3|Somme
----------------|-----------|-----------|-----------|-----------
Nb de Fissions  |36         |0          |85         |121
Nb de Collisions|29         |84         |21         |134



## Point d'entrée initial

Voyons le point d'entrée `start-init` :

`SimpleTableOutputExample4Module.cc`
\snippet SimpleTableOutputExample4Module.cc SimpleTableOutputExample4_init

Dans cet exemple, on va créer les lignes et les colonnes dans l'init au lieu
de le faire au fur et à mesure.

De plus, on va récupérer les positions des lignes.

En effet, au niveau interne, le tableau de valeur est représenté simplement
par un objet de la classe \arcane{RealUniqueArray2}, un tableau 2D.  
Niveau algorithmique, ce tableau est représenté par un tableau 1D,
chaque ligne est mise côte à côte dans la mémoire.  
Créer une ligne est donc facile (s'il y a de la place) car il suffit d'agrandir
le tableau 1D.  
Mais ajouter une colonne est sensiblement plus complexe et long car on est
obligé de décaler les valeurs de N-1 lignes. Et plus il y a de valeurs dans
le tableau, plus ce sera long.

Pour éviter tout cela, on peut créer les lignes et les colonnes dès le début.
Après, il suffira d'ajouter les valeurs où il faut.

Pour aller encore plus loin, on sauvegarde les positions des deux lignes dans
des attributs de la classe.  
Ça permet d'éviter d'effectuer une recherche de `String` dans le tableau
interne de nom des lignes `StringUniqueArray`.  
Mais c'est quelque chose de dispensable étant donné que moins il y a de lignes
moins cette recherche est couteuse et plus il y a de lignes, plus la gestion
des positions est compliqué.  
Pour cette partie, à vous de voir.


## Point d'entrée loop

Voyons le point d'entrée `compute-loop` :

`SimpleTableOutputExample4Module.cc`
\snippet SimpleTableOutputExample4Module.cc SimpleTableOutputExample4_loop

Ici, on peut voir que l'on utilise la méthode
\arcane{ISimpleTableOutput::addElementInRow()} différemment.  
En effet, cette méthode est surchargée pour permettre l'utilisation de la
position de la ligne au lieu de son nom, ce qui permet d'éviter une recherche
de String.  
De plus, cette méthode retourne un booleen qui permet de savoir si la valeur
a pu être ajouté ou non, dans le cas où la position est fausse.  
(Pour simplifier l'exemple, je ne vérifie pas la valeur retournée).


## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableOutputExample4Module.cc`
\snippet SimpleTableOutputExample4Module.cc SimpleTableOutputExample4_exit

Pour varier un peu et montrer comment utiliser les valeurs entrées dans le
tableau, on a fait la somme des deux lignes et on met ces résultats dans une
nouvelle colonne `Somme`.

\note
On n'utilise pas de ArrayView dans le service car c'est impossible de faire
une vue sur une colonne étant donné que les valeurs des colonnes sont discontinues.

Cet exemple de somme permet de voir que ce service n'est pas juste un service
d'écriture de fichier mais peut aussi stocker des valeurs pour les exploiter
après.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example3
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example5
</span>
</div>
