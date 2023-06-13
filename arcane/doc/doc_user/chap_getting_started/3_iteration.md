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
\arcane{ItemEnumerator} et possède les méthodes suivantes:

- un constructeur prenant en argument un groupe d'entité du maillage ;
- *operator++()* : pour accéder à l'élément suivant ;
- *hasNext()* : pour tester si on se trouve à la fin de l'itération ;
- _operator*()_ : qui retourne l'élément courant.

Afin d'ajouter un niveau d'abstraction supplémentaire et de
permettre d'instrumenter le code, %Arcane fournit une fonction
sous forme de macro pour chaque type d'énumérateur. Il n'est donc pas
nécessaire d'utiliser les méthodes de \arcane{ItemEnumerator}. Cette fonction
possède le prototype suivant :

```cpp
ENUMERATE_(kind, nom_iterateur, nom_groupe )
```

avec:
- **kind** le genre de l'entité (\arcane{Node}, \arcane{Cell}, ...),
- **nom_iterateur** le nom de l'itérateur
- **nom_groupe** le nom du groupe (\arcane{ItemGroup}) sur lequel on itère.

Lorsqu'on se trouve dans un module (dont la classe de base est \arcane{BasicModule})
ou un service (dont la classe de base est \arcane{BasicService}), %Arcane
fournit des méthodes pour accéder au groupe contenant toute les entités d'un genre
d'entité donné. Par exemple la méthode \arcane{BasicModule::allCells()} permet
de récupérer le groupe de toutes les mailles. Ainsi, pour itérer sur toutes les
mailles, avec **i** le nom de l'itérateur, on peut faire comme cela :

```cpp
ENUMERATE_(Cell,i,allCells())
```

La boucle de calcul de la masse décrite précédemment devient alors :

```cpp
ENUMERATE_(Cell,i,allCells()){
  m_cell_mass[i] = m_density[i] * m_volume[i];
}
```

Le type d'un énumérateur dépend du genre de l'élément de maillage : un
énumérateur sur un groupe de noeuds n'est pas du même type qu'un
énumérateur sur un groupe de mailles et ils sont donc
incompatibles. Par exemple, si la vitesse est une variable aux noeuds,
l'exemple suivant provoque une erreur de compilation :

```cpp
cout << m_velocity[i]; // Erreur!
```

De même, il est impossible d'écrire :

```cpp
ENUMERATE_(Cell,i,allNodes()) // Erreur!
```

car \arcane{BasicModule::allNodes()} est un groupe de noeud et **i** un énumérateur sur un
groupe de mailles.

Notons que l'opérateur '*' de l'énumérateur permet d'accéder à l'élément courant :
```cpp
ENUMERATE_(Cell,icell,allCells()){
  Cell cell = *icell;
}
```

Il est possible d'utiliser l'entité elle-même pour récupérer la valeur d'une variable
mais, pour des raisons de performances, il faut privilégier l'accès par l'itérateur :
```cpp
ENUMERATE_(Cell,icell,allCells()){
  Cell cell = *icell;
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
