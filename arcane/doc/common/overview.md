Vue d'ensemble de %Arcane {#arcane_overview}
=======================

[TOC]

%Arcane est un environnement de développement pour les codes de calculs
numériques parallèles. Il prend en charge les aspects architecturaux d'un code
de calcul comme les structures de données pour le maillage, le
parallélisme mais aussi des aspects plus liés à
l'environnement comme la configuration du jeu de données.

Un code de calcul, quel qu'il soit, peut être vu comme un système
prenant certaines valeurs en entrée et fournissant des valeurs en
sortie en effectuant des <em>opérations</em>. Comme il est impossible
de pouvoir traiter tous les types de codes de calcul, %Arcane se
restreint aux codes de calcul ayant les propriétés suivantes:

- le déroulement de l'exécution peut être décrit comme une
  répétition d'une suite d'opérations, chaque exécution de la suite
  d'opération étant appelé une *itération*;
- Le domaine de calcul est discrétisé dans l'espace en un ensemble
  d'éléments, le *maillage*. Ce maillage peut être 1D, 2D ou
  3D. Un maillage est constitué d'au plus quatre type d'éléments : les
  noeuds, les arêtes, les faces et les mailles. Les valeurs manipulées
  par le code reposent sur un de ces types d'éléments.

Chacun des termes décrit précédemment possède une terminologie propre
à %Arcane:

- une opération du code est appelée un [point d'entrée](@ref arcanedoc_entrypoint).
  La description de la suite d'opération s'appelle
  la [boucle en temps](@ref arcanedoc_timeloop)
- les valeurs manipulées sont appelées des [variables](@ref arcanedoc_variable)
  Par exemple, la température, la pression sont des variables.

En général, un code de calcul peut être décomposé en plusieurs parties
distinctes. Par exemple, le calcul numérique proprement dit et la
partie effectuant des sorties pour le dépouillement. De même, un code
peut utiliser plusieurs physiques : hydrodynamique, thermique, ...
Pour assurer la modularité d'un code, %Arcane fournit ce qu'on appelle
[un module](@ref arcanedoc_module) qui regroupe l'ensemble des points
d'entrée et variables correspondant à une partie donnée du code.

Enfin, les modules ont souvent besoin de capitaliser certaines
fonctionnalités. Par exemple, un module thermique et un module hydrodynamique
peuvent vouloir utiliser le même schéma numérique. Pour assurer
la capitalisation du code, %Arcane fournit ce qu'on appelle
[un service](@ref arcanedoc_service)

Les 4 notions décrites précédemment (point d'entrée, variable, module et
service) sont les notions de base de %Arcane. Elles sont décrites plus en
détail dans le document \ref arcanedoc_core. Néanmoins, avant
de voir plus en détail le fonctionnement des ces trois objets,
il faut connaître les notions de base présentées dans les chapitres
de ce document.

\note %Arcane est implémenté en utilisant le langage C++ norme 2011.
Même si des fonctionnalités avancées du C++ sont utilisés en
interne, les types et opérations mises à la disposition du développeur
numériciens ont soigneusement été étudiées pour rester le plus simple
possible.

Structures et types de base {#arcane_overview_basicstruct}
===========================

Types de bases {#arcane_overview_basicstruct_types}
--------------

%Arcane fournit un ensemble de types de base, correspondant soit à un
type existant du C++ (comme *int*, *double*), soit à une classe (comme
 Real2). Ces types sont utilisés pour toutes les opérations courantes
mais aussi pour les variables. Par exemple, lorsqu'on souhaite
déclarer un entier, il faut utiliser #Integer au lieu de
*int* ou *long*. Cela permet de modifier la taille de ces types
(par exemple, utiliser des entiers sur 8 octets au lieu de 4)
sans modifier le code source.

Les types de bases sont:

<table>
<tr><td><b>Nom de la classe</b></td><td><b>Correspondance dans les spécifications</b></td></tr>
<tr><td>#Integer   </td><td> entier signé </td></tr>
<tr><td>#Int16     </td><td> entier signé sur 16 bits </td></tr>
<tr><td>#Int32     </td><td> entier signé sur 32 bits </td></tr>
<tr><td>#Int64     </td><td> entier signé sur 64 bits </td></tr>
<tr><td>#Byte      </td><td> représente un caractère sur 8 bits </td></tr>
<tr><td>#Real      </td><td> réel IEEE 754 </td></tr>
<tr><td>Real2     </td><td> coordonnée 2D, vecteur de deux réels </td></tr>
<tr><td>Real3     </td><td> coordonnée 3D, vecteur de trois réels </td></tr>
<tr><td>Real2x2   </td><td> tenseur 2D, vecteur de quatre réels </td></tr>
<tr><td>Real3x3   </td><td> tenseur 3D, vecteur de neufs réels </td></tr>
<tr><td>String    </td><td> chaîne de caractères ISO-8859-1 </td></tr>
</table>

Par défaut, les entiers (#Integer) sont stockés sur 4 octets mais il
est possible de passer sur 8 octets en compilant avec la macro
**ARCANE_64BIT**. Par défaut, les flottants (#Real, Real2, Real2x2,
Real3, Real3x3) utilisent des réels double précision de la normae IEEE
754 et sont stockés sur 8 octets.

Entités du maillage {#arcane_overview_basicstruct_meshitem}
-------------------

Il existe 4 types d'entités de base dans un maillage : les noeuds, les
arêtes, les faces et les mailles. A chacun de ces types correspond une
classe C++ dans %Arcane. Pour chaque type d'entité, il existe un type
*groupe* qui gère un ensemble d'entités de ce type. La classe qui gère
un groupe d'une entité a pour nom celui de l'entité suffixé par
*Group*. Par exemple, pour les noeuds, il s'agit de #NodeGroup.

<table>
<tr><td><b>Nom de la classe</b></td><td><b>Correspondance dans les spécifications</b></td></tr>
<tr><td>Node      </td><td> un noeud </td></tr>
<tr><td>Face      </td><td> une face en 3D, une arête en 2D</td></tr>
<tr><td>Edge      </td><td> une arête en 3D</td></tr>
<tr><td>Cell      </td><td> une maille </td></tr>
<tr><td>#NodeGroup </td><td> un groupe de noeuds </td></tr>
<tr><td>#EdgeGroup </td><td> un groupe d'arêtes </td></tr>
<tr><td>#FaceGroup </td><td> un groupe de faces </td></tr>
<tr><td>#CellGroup </td><td> un groupe de mailles </td></tr>
</table>

\note
En dimension 2, les faces correspondent aux arêtes et en dimension 3 aux
faces. Ceci permet aux algorithmes numériques de parcourir le maillage
indépendamment de sa dimension. L'entité arête (Edge) n'existe que
pour les maillages 3D et correspond alors à une arête.

Chaque entité du maillage correspond à une instance d'une classe. Par
exemple, si le maillage contient 15 mailles, il y a 15 instances du
type Cell. Chaque classe fournit un certain nombre d'opérations
permettant de relier les instances entre elles. Par exemple, la méthode
Cell::node(Integer) de la classe *Cell* permet de récupérer le ième
noeud de cette maille. De même, la méthode Cell::nbNode() permet de
récupérer le nombre de noeuds de la maille. Pour plus de
renseignements sur les opérations supportées, il est nécessaire de se
reporter à la documentation en ligne des classes correspondantes
(Node, Edge, Face, Cell).

Il existe d'autres types d'entités comme les particules (Particle),
les liens (Link), les noeuds duaux (DualNode) ou les degrés de liberté
(DoF).

Itération {#arcane_overview_iteration}
=========

Avant de pouvoir coder une opération, il
faut bien comprendre comment s'écrit une boucle sur une liste
 d'entités de maillage telles que les mailles ou les noeuds. En effet, pratiquement toutes les opérations
que l'on effectue se font sur un ensemble d'entités et donc
comportent une boucle sur une liste d'entités. Par exemple, calculer
la masse des mailles consiste à boucler sur l'ensemble des mailles et
pour chacune d'elle effectuer le produit de son volume par sa
densité. Conventionnellement, cela peut s'écrire de la manière
suivante :

~~~~~~~~~~~~~~~~{.cpp}
for( Integer i=0; i<nbCell(); ++i )
  m_cell_mass[i] = m_density[i] * m_volume[i];
~~~~~~~~~~~~~~~~

La boucle *for* comprend trois parties séparées par un
point-virgule. La première est l'initialisation, la seconde est le
test de sortie de boucle et la troisième est l'opération effectuée
entre deux itérations.

L'écriture précédente a plusieurs inconvénients :
- elle fait apparaître la structure de donnée sous-jacente, à
  savoir un tableau;
- elle utilise un indice de type entier pour accéder aux éléments.
  Ce typage faible est source d'erreur car il ne permet pas, entre autre,
  de tenir compte du genre de la variable. Par exemple, on pourrait
  écrire \c m_velocity[i] avec \c i étant un numéro de maille et
  \c m_velocity une variable aux noeuds;
- elle oblige à ce que la numérotation des entités soit contigue.

En considérant qu'on parcourt toujours la liste des entités dans le
même ordre, il est possible de modéliser le comportement précédent par
quatre opérations :

- initialiser un compteur au début du tableau;
- incrémenter le compteur;
- regarder si le compteur est à la fin du tableau;
- retourner l'élément correspondant au compteur.

Le mécanisme est alors général et indépendant du type du conteneur :
l'ensemble des entités pourrait être implémenté sous forme de
tableau ou de liste sans changer ce formalisme. Dans l'architecture,
le compteur ci-dessus est appelé un *itérateur* et itérer sur
l'ensemble des éléments se fait en fournissant un itérateur de début
et de fin, autrement appelé un *énumérateur*

Dans %Arcane, cet énumérateur dérive de la classe de base
ItemEnumerator et possède les méthodes suivantes:

- un constructeur prenant en argument un groupe d'entité du maillage;
- *operator++()*: pour accèder à l'élément suivant;
- *hasNext()* : pour tester si on se trouve à la fin de l'itération;
- _operator*()_: qui retourne l'élément courant.

Afin d'ajouter un niveau d'abstraction supplémentaire et de
permettre d'instrumenter le code, %Arcane fournit une fonction
sous forme de macro pour chaque type d'énumérateur. Cette fonction
possède le prototype suivant:

~~~~~~~~~~~~~~~~{.cpp}
ENUMERATE_[type]( nom_iterateur, nom_groupe )
~~~~~~~~~~~~~~~~

avec:
- **[type]** le type d'élément (\c NODE, \c CELL, ...),
- **nom_iterateur** le nom de l'itérateur
- **nom_groupe** le nom du groupe sur lequel on itère.

Par exemple, pour itérér sur toutes les mailles, avec **i** le nom de l'itérateur:

~~~~~~~~~~~~~~~~{.cpp}
ENUMERATE_CELL(i,allCells())
~~~~~~~~~~~~~~~~

La boucle de calcul de la masse décrite précédemment devient alors :

~~~~~~~~~~~~~~~~{.cpp}
ENUMERATE_CELL(i,allCells()){
  m_cell_mass[i] = m_density[i] * m_volume[i];
}
~~~~~~~~~~~~~~~~

Le type d'un énumérateur dépend du type d'élément de maillage : un
énumérateur sur un groupe de noeuds n'est pas du même type qu'un
énumérateur sur un groupe de mailles et ils sont donc
incompatibles. Par exemple, si la vitesse est une variable aux noeuds,
l'exemple suivant provoque une erreur de compilation:

~~~~~~~~~~~~~~~~{.cpp}
cout << m_velocity[i]; // Erreur!
~~~~~~~~~~~~~~~~

De même, il est impossible d'écrire :

~~~~~~~~~~~~~~~~{.cpp}
ENUMERATE_CELL(i,allNodes()) // Erreur!
~~~~~~~~~~~~~~~~

car **allNodes()** est un groupe de noeud et **i** un énumérateur sur un
groupe de mailles.

Notons que l'opérateur '*' de l'énumérateur permet d'accéder à l'élément courant:
~~~~~~~~~~~~~~~~{.cpp}
ENUMERATE_CELL(i,allCells()){
  Cell cell = *i;
}
~~~~~~~~~~~~~~~~

Il est possible d'utiliser l'entité elle-même pour récupérer la valeur d'une variable
mais, pour des raisons de performances, il faut privilégier l'accès par l'itérateur:
~~~~~~~~~~~~~~~~{.cpp}
ENUMERATE_CELL(icell,allCells()){
  Cell cell = *i;
  m_cell_mass[cell] = m_density[cell] * m_volume[cell]; // moins performant
  m_cell_mass[icell] = m_density[icell] * m_volume[icell]; // plus performant
}
~~~~~~~~~~~~~~~~
