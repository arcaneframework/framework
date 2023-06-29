# Optimisations de la connectivité des entités {#arcanedoc_new_optimisations_connectivity}

[TOC]

## Contexte

Dans la version actuelle de %Arcane (3.x), les
connectivités des entités et des groupes d'entités sont gérés comme si
tout était non-structuré même si le maillage est cartésien ou
structuré. Notamment, on conserve toutes les informations de
connectivités entre les entités ce qui consomme de la mémoire. Ce coût
mémoire est difficilement réductible lorsque le maillage est vraiment
non structuré, mais il pourrait l'être pour les maillages
structurés et cartésiens. Particulièrement pour ces derniers cas, on a souvent
beaucoup de mailles et peu de variables et proportionnellement ce cout
mémoire de conservation des connectivités est important.

Il a donc été décidé de modifier la gestion interne à %Arcane de ces
informations de connectivité afin de réduire la consommation mémoire
et de permettre des optimisations supplémentaires.

Ces optimisations doivent répondre aux contraintes suivantes :

- Impacter le moins possible les codes utilisant %Arcane et donc ces
  évolutions devront être progressive pour laisser les codes faire les
  évolutions nécessaires (comme cela a été le cas avec les nouvelles
  connectivités pour la version 3.0).
- le mécanisme de gestion des connectivités doit rester générique pour
  s'appliquer à tous les types de maillage avec les mécanismes
  actuels de boucle et d'itération.

## Optimisations envisagées

Les optimisations envisagées partent du principe que le schéma des
connectivités et des groupes est souvent le même pour un grand nombre
d'entités. Les optimisations proposées se décomposent en deux groupes :

- les optimisations sur les connectivités (par exemple \arcane{Cell::nodes()})
- les optimisations sur les groupes d'entités (\arcane{ItemGroup}).

### Optimisations des connectivités

Actuellement, les connectivités sont gérées par la classe
\arcane{mesh::IncrementalItemConnectivity} et il y a trois tableaux (de \arcane{Int32})
pour conserver la connectivité d'une entité vers une autre :

1. un tableau contenant la connectivité. La taille de ce tableau est
au moins égal à la somme des entités connectées. Par exemple si on a 40
hexaèdres, alors sa taille est de 8x40 éléments.
2. un tableau par entité indiquant combien elle a d'entités connectées.
3. un tableau par entité indiquant la position dans le tableau (1) du
premier élément connecté.

L'optimisation proposée part du principe que pour une entité donnée les `localId()`
des entités qui lui sont connectées sont souvent les mêmes relativement au
`localId()` de la première entité connectée. Au lieu de conserver toute
la connectivité, on peut donc ne conserver que le schéma de la
connectivité ainsi que le localId() de la première entité.

Pour cette optimisation, il faut donc :
1. conserver pour chaque entité le localId() de la première entité
connectée
2. lors de l'accès à la i-ème entité connecté, il faut ajouter à la
valeur conservée le localId() de la première entité connectée.

L'opération (2) aura un cout négligeable en temps de calcul, car la
valeur à ajouter sera conservée lors de l'itération de la même manière
que le nombre d'entités connectées. Il reste donc pour (1) l'ajout
d'un 'Int32' pour chaque entité, mais cela sera compensé par la
réutilisation de la connectivité.

Par exemple, pour un maillage cartésien de 2 lignes et 4
colonnes, on a actuellement la numérotation suivante :

```
10---11---12---13---14
| 4  | 5  | 6  | 7  |
5----6----7----8----9
| 0  | 1  | 2  | 3  |
0----1----2----3----4
```

Si je prends la connectivité maille/nœuds, les trois tableaux
contiennent les valeurs suivantes :

```
1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
2. 4         4         4         4         4           4           4           4
3. 0         4         8        12         16          20         24          28
```

Si j'applique l'optimisation, j'ajoute le tableau suivant contenant le
localId() de la première entité :

```
1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
2. 4         4         4         4         4           4           4           4
3. 0         4         8        12         16          20         24          28
4. 0         1         2         3         5           6           7           8
```

Je retranche de (1) la valeur associée de (4)

```
1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
4. 0         1         2         3         5           6           7           8

1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
-  0 0 0 0   1 1 1 1   2 2 2 2   3 3 3 3   5 5  5  5   6  6  6 6   7 7  7  7 | 8  8  8 8
=  0 5 6 1   0 5 6 1   0 5 6 1   0 5 6 1   0 5  6  1   0  5  6 1   0 5  6  1   0  5  6 1
```

On voit donc que pour ce cas (idéal), le schéma est le même. On peut donc ne le conserver
qu'une fois et on aura les valeurs suivantes pour la connectivité :

```
1. 0 5 6 1
2. 4         4         4         4         4           4           4           4
3. 0         0         0         0         0           0           0           0
4. 0         1         2         3         5           6           7           8
```

Si `N` est le nombre de mailles, on passe donc de d'une consommation mémoire
de `(N*4 + N + N)` à `(4 + N + N +N)` soit de `6*N` à `3*N`. Dans le cas 3D,
on passe de `(N*8 + N + N)` à `(8 + N + N + N)` soit `10*N` à `3*N`.

\note On pourrait aussi envisager de conserver le nombre d'entités
connectées à une entité dans (1) ce qui permettrait de supprimer le
tableau (2).

Dans les maillages cartésiens, le schéma pour les mailles et les
nœuds est indépendant du nombre de mailles et de noeuds. Par contre,
pour les faces, il y a un schéma par ligne et un par colonne. Donc par
exemple pour un maillage 100x30x20 il y a 32x20 schémas pour les
faces soit 640 valeurs au lieu de 60000 sans l'optimisation.

Outre une réduction de la consommation mémoire cela permettra de mieux
utiliser le cache.

Dans le pire des cas s'il n'y a pas de schéma récurrent, la
connectivité consommera `N` \arcane{Int32} supplémentaires. A priori cela n'est
le cas que pour les maillages composés de triangles (2D) ou tétraèdres
(3D) quelconques ce qui n'est pas le cas des applications CEA et IFPEN.

Ces mécanismes peuvent aussi s'appliquer aux classes gérant
spécifiquement le cartésien qui ont aussi des schémas d'accès
similaires.

### Démarche pour mettre en place ces optimisations

Afin de pouvoir procéder à ces optimisations de manière transparente
il faut un itérateur sur les connectivités différent de l'itérateur
sur les entités ce qui n'est pas le cas actuellement, car les deux
utilisent \arcane{ItemVectorView} et \arcane{ItemEnumerator} comme conteneur et
itérateur.
Cela permet de coder comme suit :

```{cpp}
Arcane::CellGroup cells;
ENUMERATE_(Cell,icell,cells){
  Arcane::Cell cell = *icell;
  ENUMERATE_(Face,iface,cell.faces()){
  }
}
```

Cela devra donc être interdit pour les connectivités. On pourra
ajouter une macro spécifique pour énumérer sur les connectivités ou
aussi remplacer par le `for-loop` :

```{cpp}
Arcane::CellGroup cells;
ENUMERATE_(Cell,icell,cells){
  Arcane::Cell cell = *icell;
  // for-loop
  for ( Arcane::Face face : cell.faces()){
  }
}
```

### Optimisations des groupes

Le même principe que pour les connectivités peut être appliqué aux groupes.
Actuellement, on conserve une liste d'indirection simple. Pour un
groupe avec M éléments, il faut conserver M \arcane{Int32}.

Il serait possible de décomposer la liste des entités en blocs et
conserver pour chaque bloc 3 valeurs correspondantes aux tableaux (2),
(3) et (4) des connectivités. Éventuellement le nombre d'éléments dans
chaque bloc peut aussi être mutualisé dans la liste d'indirection.

Avec par exemple une taille de bloc de 128, il faut conserver
(3*M/128) valeurs pour les informations d'indirection. Mais dans le
cas où les valeurs du tableau sont contigues, il faut juste conserver
en plus 128 valeurs. La encore cela permet d'économiser de la mémoire
et de mieux utiliser le cache. Ce mécanisme a aussi l'avantage d'être
facilement utilisable sur accélérateur.

### Démarche pour mettre en place ces optimisations

Cela nécessite cependant deux modifications dans %Arcane:
- rendre obsolète la possibilité de récupérer les `localIds()` des
groupes
- transformer la macro `ENUMERATE_` pour faire deux boucles : une
boucle sur les blocs suivie d'une boucle pour chaque bloc. Cela
nécessite donc de changer la classe \arcane{ItemEnumerator} pour gérer les
blocs ou d'en faire une autre. Une possibilité est par exemple
d'avoir deux nouvelles classes `ItemBlockVectorView` et
`ItemBlockEnumerator`. Le cas actuel utilisant \arcane{ItemVectorView} et
\arcane{ItemEnumerator} serait un cas spécifique de bloc dont le nombre de
valeurs correspondrait au nombre d'éléments du vecteur et l'offset de
`localId()` serait zéro.

Il doit être possible d'utiliser toujours une double boucle dans
`ENUMERATE_` quitte à ce que la deuxième boucle soit de taille fixée à
la compilation à 1 dans le cas des \arcane{ItemVectorView} par exemple.

### Planning

Les modifications pour changer ces connectivités commenceront dans la
version 3.10 de %Arcane (juin 2023).

La première optimisation envisagée concernerait les connectivités des
entités. Dans ce but, les modifications suivantes sont effectuées :

1. les méthodes permettant d'accéder aux tableaux de `localId()` des
   connectivités seront rendues obsolètes. Cela concerne les classes
   \arcane{ItemEnumerator}, \arcane{ItemEnumeratorBase},
   \arcane{ItemConnectedListView}.
2. L'utilisation de la classe \arcane{ItemInternal} devient
   obsolète. Il faut utiliser \arcane{impl::ItemBase} à la
   place. Toutes les méthodes qui prenaient aussi des
   \arcane{ItemInternalArrayView}, \arcane{ItemInternalPtr} ou
   \arcane{ItemInternalList} deviennent aussi obsolètes.

En général les codes utilisateurs de %Arcane sont peu impactés par le
point (1) car les structures concernées sont plutôt internes à
Arcane. Comme il est nécessaire de supprimer ces méthodes pour mettre
en place connectivités compressées, il est prévu de supprimer
définitivement ces méthodes en décembre 2023.

Pour le point (2) qui concerne potentiellement plus de parties de
code, il est prévu de supprimer les méthodes concernées en juin 2024.

La deuxième phase d'optimisation concernant les entités des groupes
interviendra ensuite.

