# Itération {#arcanedoc_getting_started_iteration}

[TOC]

Avant de pouvoir coder une opération, il
faut bien comprendre comment s'écrit une boucle sur une liste
 d'entités de maillage telles que les mailles ou les noeuds. En effet, pratiquement toutes les opérations
que l'on effectue se font sur un ensemble d'entités et donc
comportent une boucle sur une liste d'entités. Par exemple, calculer
la masse des mailles consiste à boucler sur l'ensemble des mailles et
pour chacune d'elle effectuer le produit de son volume par sa
densité. Conventionnellement, cela peut s'écrire de la manière
suivante :

```cpp
for( Integer i=0; i<nbCell(); ++i )
  m_cell_mass[i] = m_density[i] * m_volume[i];
```

La boucle *for* comprend trois parties séparées par un
point-virgule. La première est l'initialisation, la seconde est le
test de sortie de boucle et la troisième est l'opération effectuée
entre deux itérations.

L'écriture précédente a plusieurs inconvénients :
- elle fait apparaître la structure de donnée sous-jacente, à
  savoir un tableau ;
- elle utilise un indice de type entier pour accéder aux éléments.
  Ce typage faible est source d'erreur car il ne permet pas, entre autre,
  de tenir compte du genre de la variable. Par exemple, on pourrait
  écrire \c m_velocity[i] avec \c i étant un numéro de maille et
  \c m_velocity une variable aux noeuds ;
- elle oblige à ce que la numérotation des entités soit contigüe.

En considérant qu'on parcourt toujours la liste des entités dans le
même ordre, il est possible de modéliser le comportement précédent par
quatre opérations :

- initialiser un compteur au début du tableau ;
- incrémenter le compteur ;
- regarder si le compteur est à la fin du tableau ;
- retourner l'élément correspondant au compteur.

Le mécanisme est alors général et indépendant du type du conteneur :
l'ensemble des entités pourrait être implémenté sous forme de
tableau ou de liste sans changer ce formalisme. Dans l'architecture,
le compteur ci-dessus est appelé un *itérateur* et itérer sur
l'ensemble des éléments se fait en fournissant un itérateur de début
et de fin, autrement appelé un *énumérateur*

Dans %Arcane, cet énumérateur dérive de la classe de base
ItemEnumerator et possède les méthodes suivantes :

- un constructeur prenant en argument un groupe d'entité du maillage ;
- *operator++()* : pour accéder à l'élément suivant ;
- *hasNext()* : pour tester si on se trouve à la fin de l'itération ;
- _operator*()_ : qui retourne l'élément courant.

Afin d'ajouter un niveau d'abstraction supplémentaire et de
permettre d'instrumenter le code, %Arcane fournit une fonction
sous forme de macro pour chaque type d'énumérateur. Cette fonction
possède le prototype suivant :

```cpp
ENUMERATE_[type]( nom_iterateur, nom_groupe )
```

avec:
- **[type]** le type d'élément (\c NODE, \c CELL, ...),
- **nom_iterateur** le nom de l'itérateur
- **nom_groupe** le nom du groupe sur lequel on itère.

Par exemple, pour itérer sur toutes les mailles, avec **i** le nom de l'itérateur :

```cpp
ENUMERATE_CELL(i,allCells())
```

La boucle de calcul de la masse décrite précédemment devient alors :

```cpp
ENUMERATE_CELL(i,allCells()){
  m_cell_mass[i] = m_density[i] * m_volume[i];
}
```

Le type d'un énumérateur dépend du type d'élément de maillage : un
énumérateur sur un groupe de noeuds n'est pas du même type qu'un
énumérateur sur un groupe de mailles et ils sont donc
incompatibles. Par exemple, si la vitesse est une variable aux noeuds,
l'exemple suivant provoque une erreur de compilation :

```cpp
cout << m_velocity[i]; // Erreur!
```

De même, il est impossible d'écrire :

```cpp
ENUMERATE_CELL(i,allNodes()) // Erreur!
```

car **allNodes()** est un groupe de noeud et **i** un énumérateur sur un
groupe de mailles.

Notons que l'opérateur '*' de l'énumérateur permet d'accéder à l'élément courant :
```cpp
ENUMERATE_CELL(i,allCells()){
  Cell cell = *i;
}
```

Il est possible d'utiliser l'entité elle-même pour récupérer la valeur d'une variable
mais, pour des raisons de performances, il faut privilégier l'accès par l'itérateur :
```cpp
ENUMERATE_CELL(icell,allCells()){
  Cell cell = *i;
  m_cell_mass[cell] = m_density[cell] * m_volume[cell]; // moins performant
  m_cell_mass[icell] = m_density[icell] * m_volume[icell]; // plus performant
}
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_getting_started_basicstruct
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
