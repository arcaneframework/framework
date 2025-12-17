# Gestion de la géométrie {#arcanedoc_entities_geometric}

[TOC]

Cette page décrit les classes gérant la géométrie dans %Arcane.

Le but des classes géométriques de %Arcane est de fournir une
structure de donnée unifiée pour gérer de manière efficace les
opérations sur les polygones (en 2D) et les polyèdres (en 3D).

## Introduction {#arcanedoc_entities_geometric_introduction}

Les polygones et polyèdres supportés sont définis dans l'énumération
GeomType. Il s'agit de :
- en 2D, les triangles, les quadrangles, les pentagones et les
hexagones.
- en 3D, les tétraèdres, les pyramides, les prismes classiques, les
hexaèdres, les prismes à base pentagonales (heptaèdres) et les
prismes à base hexagonales (octaèdres).

%Arcane fournit deux types d'objets pour gérer la géométrie.
- le premier type et le plus simple est appelé un <b>élément
géométrique</b> et contient uniquement les coordonnées des noeuds de
cet élément. Ces classes ont pour nom le type de l'élément suivi de
\b %Element. Par exemple, pour les hexaèdres, le nom est Hexaedron8Element.
- le second type est appelé une <b>forme géométrique</b> et contient en
plus des coordonnées des noeuds, les coordonnées des centres des
faces, des milieux des arêtes et du centre ainsi que des informations
sur la connectivité. Les formes géométriques sont gérées par la
classe GeomShape et les vues sur ces formes géométriques par la
classe GeomShapeView (anciennement GenericElement). En général, seule
la vue est utilisée.

## Éléments géométriques {#arcanedoc_entities_geometric_geomelement}

Le terme <b>élément géométrique</b> englobe l'ensemble des classes
qui gèrent des éléments géométriques en conservant uniquement les
coordonnées des sommets de ces éléments. Les différentes classes
sont :
- Triangle3Element;
- Quad4Element;
- Pentagon5Element;
- Hexagon6Element;
- Tetraedron4Element;
- Pyramid5Element;
- Pentaedron6Element;
- Hexaedron8Element;
- Heptaedron10Element;
- Octaedron12Element;

Elles s'utilisent de la même manière, seul le nombre de coordonnées
différe :
```cpp
Real3 x0,x1,x2,x3,x4,x5,x6,x7,x8;
Quad4Element quad(x0,x1,x2,x3); // Création d'un quad avec initialisation
Hexaedron8Element hexa; // Création d'un hexa non initialisé
hexa.init(x0,x1,x2,x3,x4,x5,x6,x7); // Initialisation
hexa[5] = Real3(1.2,0.0,0.0); // Change la valeur du 6-ème sommet
Real3 z = hexa[4]; // Récupère la valeur du 5-ème sommet
```

Les éléments géométriques s'utilisent en général via la notion de
vue, à la manière des classes tableaux (Array, ArrayView et
ConstArrayView). Il existe donc une vue modifiable et une vue
constante pour chaque type d'élément géométrique. Pour obtenir le nom
de la vue, il suffit d'ajouter \a View ou \a ConstView au nom de la
classe:
```cpp
Quad4Element quad;
Quad4ElementView quad_view = quad.view();
Quad4ElementConstView quad_const_view = quad.constView();
```

La conversion d'un élément vers une vue peut se faire automatiquement :

```cpp 
Quad4Element quad;
Quad4ElementView quad_view = quad;
Quad4ElementConstView quad_const_view = quad;
```

## Formes géométriques {#arcanedoc_entities_geometric_geomshape}

Contrairement aux éléments géométriques, il n'existe qu'une seule
classe pour gérer les formes géométriques. Cette classe s'appelle
GeomShape et peut contenir les informations géométriques de n'importe
quel type de maille définie dans l'énumération GeomType.

Une forme géométrique contient les coordonnées des noeuds, du centre
des faces, du milieu des arêtes et du centre de la forme.

La forme géométrique s'utilise exclusivement via une vue sur une
GeomShape. Cette vue est appelée GeomShapeView et contient toutes les
méthodes pour récupérer les informations nécessaires sur la forme
géométrique. Il existe aussi des vues spécifiques par type
géométrique. Comme pour les vues sur les éléments géométriques, ces
classes ont pour nom le type géométrique suffixé par \a ShapeView:
- Triangle3ShapeView;
- Quad4ShapeView;
- Pentagon5ShapeView;
- Hexagon6ShapeView;
- Tetraedron4ShapeView;
- Pyramid5ShapeView;
- Pentaedron6ShapeView;
- Hexaedron8ShapeView;
- Heptaedron10ShapeView;
- Octaedron12ShapeView;

%Arcane gère deux utilisations possibles des formes géométriques :
- la forme géométrique associée à une maille du maillage. Pour cet
usage, on utilise la classe GeomShapeMng qui conserve pour un
maillage donné l'ensemble des informations nécessaires (voir la
documentation de la classe GeomShapeMng pour son utilisation et son
initialisation). La récupération d'une vue se fait comme suit :
```cpp
GeomShapeMng& shape_mng;
Cell cell;
GeomShapeView shape_view;
// Initialisation à partir d'une maille \a cell
shape_mng.initShape(shape_view,cell);
```

- la forme géométrique quelconque, qui n'est pas directement associée
à une entité du maillage et qui peut être créée n'importe où. Elle
peut être utilisée par exemple pour définir une forme géométrique sur
les sous-volumes de contrôle d'une maille. Pour ce cas, il faut
utiliser une instance de GeomShape pour conserver les
informations. Cette instance doit rester valide tant qu'on souhaite
utiliser la vue qui lui est associée. L'initialisation se fait soit
avec un hexaèdre, soit avec un quadrangle (il est prévu dans une
version ultérieure de pouvoir initialiser avec d'autres types). Par
exemple, pour un hexaèdre :
```cpp
GeomShape shape;
Hexaedron8Element hexa;
// Initialisation à partir d'un élément géométrique existant \a hexa.
Hexaedron8ShapeView shape_view = shape.initFromHexaedron8(hexa);
```

## Utilisation des vues {#arcanedoc_entities_geometric_viewusage}

\warning Comme toutes les classes qui utilisent la notion de vue dans
%Arcane, les vues sur les objets géométriques ne restent valides que
tant que le conteneur desquelles elles sont issues reste valide. En
particulier, il faut restreindre leur utilisation au passage de
paramètres entre méthodes et il ne faut <b>JAMAIS</b> conserver une
vue au cours d'un calcul (comme champ d'une classe par exemple).

Lorsqu'on souhaite utilise un objet géométrique, le type des vues à
utiliser dépend des coordonnées dont on a besoin :
- si on a besoin uniquement des coordonnées des noeuds de l'élément,
il faut utiliser une vue sur un élément. La vue doit être constante
si on ne modifie pas l'élément. Par exemple, pour une méthode qui
calcule le volume d'un hexaèdre, il faut utiliser un
Hexaedron8ElementConstView comme paramètre.
- si on a besoin en plus des coordonnées des noeuds, des coordonnées
des centres des faces, du milieu des arêtes ou du centre de
l'élément, il faut utiliser une vue sur une forme. Si on ne connait
pas exactement le type de la forme, il faut utiliser un
GeomShapeView. Si on connait le type exact, il faut utiliser la vue
correspondante. Par exemple, pour un quadrangle, un Quad4ShapeView.

Il existe aussi une class GeomShapeOperation qui permet d'obtenir une
classe qui implémente IItemOperationByBasicType en fournissant
uniquement les opérations pour une type de vue données (voir la
documentation de GeomShapeOperation pour plus d'informations)

### Utilisation des vues sur les éléments géométriques. {#arcanedoc_entities_geometric_geomelementview}

Les vues sur les éléments doivent être utilisées partout où c'est
possible, notamment au lieu de passer \a N coordonnées en
argument. Par exemple, pour une méthode de calcul de la surface d'un
quadrangle, au lieu de :
```cpp
Real computeSurface2D(const Real3& a0, const Real3& a1,
                      const Real3& a2, const Real3& a3)
{  
  Real3 fst_diag = a2 - a0;
  Real3 snd_diag = a3 - a1;
  return 0.5 * math::crossProduct2D(fst_diag,snd_diag);
}
```

il vaut mieux utiliser :

```cpp
Real computeSurface2D(Quad4ElementConstView quad)
{  
  Real3 fst_diag = quad[2] - quad[0];
  Real3 snd_diag = quad[3] - quad[1];
  return 0.5 * math::crossProduct2D(fst_diag,snd_diag);
}
```

L'intérêt d'utiliser la vue est multiple :
- Les éléments et formes géométriques peuvent facilement être
convertibles en une vue :
```cpp
// Utilisation avec un élément géométrique.
Quad4Element my_quad;
computeSurface2D(my_quad);

// Utilisation avec une forme géométrique
GeomShapeMng& shape_mng;
Cell cell;
GeomShapeView shape_view;
shape_mng.initShape(shape_view,cell);
computeSurface2D(shape_view.toQuad4Element());
```
- il est possible de créer une vue à partir de \a N coordonnées :
```cpp
// Utilisation à partir de 4 réels
Real3 a0,a1,a2,a3;
computeSurface2D(Quad4Element(a0,a1,a2,a3));

// Utilisation à partir d'un tableau de 4 réels
Real3 a[4];
computeSurface2D(Quad4Element(Real3ConstArrayView(4,a)));

// Utilisation à partir des coordonnées d'une entité.
VariableNodeReal3& node_coords;
Face face;
computeSurface2D(Quad4Element(node_coords,face));
```
- possibilité de spécialiser les opérations lorsqu'on utilise des
* templates et notamment de faire la distinction entre les
* opérations qui prennent le même nombre de coordonnées mais qui
* opèrent sur des éléments différents (par exemple entre un Quad4 et
* un Tetraedron4)
- à terme, possibilité de vectoriser plus facilement.

La vue sur les éléments géométriques permet donc d'unifier plusieurs
mécanismes d'appels et doit donc être utilisée dans tous les cas où
cela est possible (i.e. c'est-à-dire tout le temps). En
particulier, les méthodes qui prennent \a N coordonnées en argument
peuvent toujours être remplacées par une méthode qui prend une vue de
l'élément correspondant. 

### Utilisation des GeomShapeView {#arcanedoc_entities_geometric_geomshapeview}

Les GeomShapeView sont optimisées pour les calculs géométriques au
sein d'une maille. Il est donc préférable de les utiliser plutôt que
d'aller chercher à chaque fois les coordonnées des noeuds d'une
maille en passant par la variable IMesh::nodesCoordinates(). En
particulier, elles utilisent une structure de donnée qui est
optimisée pour la gestion du cache et pour la vectorisation. De plus,
elles permettront à terme de gérer des formes géométriques
correspondantes à des éléments finis d'ordre 2 ou supérieur.

Par exemple, pour récupérer le milieu des noeuds 3 et 4 d'une maille :
```cpp
// Méthode classique
VariableNodeReal3& node_coord = ...;
ENUMERATE_CELL(icell,allCells()){
  Cell cell = *icell;
  Real3 middle = (node_coord[cell.node(3)] + node_coord[cell.node(4)]) / 2.0;
}

// Méthode optimisée.
GeomShapeMng& shape_mng = ...;
GeomShapeView shape;
ENUMERATE_CELL(icell,allCells()){
  shape_mng.initShape(shape,*icell);
  Real3 middle = (shape.node(3) + shape.node(4)) / 2.0;
}
```



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_itemtype
</span>
<span class="next_section_button">
\ref arcanedoc_entities_tools
</span>
</div>
