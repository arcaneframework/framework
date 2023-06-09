# Modifications dans la gestions des entités {#arcanedoc_item_handling_news}

Cette page regroupe les modifications relatives à la gestion des connectivités
des entités lors des différentes versions de %Arcane.

[TOC]

## Modifications dans la version 3.7 {#arcanedoc_news_37}

La version 3.7 de %Arcane apporte plusieurs modifications dans la
gestion des entités du maillage. Ces modifications ont pour objectif:

- de réduire l'empreinte mémoire
- de pouvoir accéder sur accélérateurs à certaines informations sur
  les entités comme par exemple Arcane::Item::owner().
- d'améliorer les performances

Pour répondre à ces objectifs, les modifications suivantes ont été
apportées:

- La classe Arcane::ItemSharedInfo ne contient plus d'informations
  spécifiques sur une entité. Il n'y a donc besoin que d'une seule
  instance de cette classe par Arcane::IItemFamily. La méthode
  Arcane::mesh::ItemFamily::commonItemSharedInfo() permet de récupérer
  cette instance.
- Comme il n'y a plus qu'une seule instance de Arcane::ItemSharedInfo,
  l'utilisation de Arcane::ItemInternal devient optionnelle pour créer
  une instance de Arcane::Item ou d'une des classes dérivées. Il est
  maintenant possible de créer une instance de ces classes uniquement
  avec un Arcane::ItemLocalId et un Arcane::ItemSharedInfo*. La classe
  interne Arcane::ItemBaseBuildInfo est utilisée pour cela.
- Une classe Arcane::ItemBase a été créée. Elle contient uniquement un
  Arcane::ItemLocalId et un Arcane::ItemSharedInfo*. Elle sert de
  classe de base à Arcane::Item et Arcane::ItemInternal. Cette classe
  est interne à Arcane mais ne permet pas de modifier les valeurs
  d'une entité (par exemple elle contient une méthode
  Arcane::ItemBase::owner() mais pas de méthode `setOwner()`). Grâce à
  cette classe, Arcane::Item ne dépend plus de Arcane::ItemInternal et
  peut directement récupérer les informations via
  Arcane::ItemSharedInfo ce qui éviter un niveau d'indirection pour
  accéder aux connectivités par exemple.
- Les itérateurs sur les entités (Arcane::ItemEnumerator) ont été
  modifiés pour qu'ils n'utilisent plus Arcane::ItemInternal à chaque
  incrémentation ce qui peut légèrement améliorer les performances car
  on évite une indirection supplémentaire à chaque fois.
- les données qui étaient portées par Arcane::ItemInternal (comme
  `owner()`, `flags()`) sont maintenant conservées par des variables
  tableaux (Arcane::VariableArrayInt32 ou Arcane::VariableArrayInt16)
  gérées par la famille d'entité.

Avec ces modifications, à terme il sera possible de supprimer
entièrement l'utilisation de Arcane::ItemInternal.

Cependant, cette classe est souvent utilisée dans les codes donc ce
changement doit être progressif. Notamment, la méthode
Arcane::IItemFamily::itemsInternal() est utilisée pour récupérer une
instance de Arcane::Item à partir d'un Arcane::ItemInternalArrayView
(qui est le type retourné par cette méthode).

Pour préparer cela et garder le code compatible, une nouvelle classe
Arcane::ItemInfoListView (et les classes dérivées spécifiques aux
entités comme Arcane::CellInfoListView, Arcane::DoFInfoListView)
permet de récupérer les informations sur les entités pour lesquelles
il fallait auparavant utiliser Arcane::IItemFamily::itemsInternal().

Il est possible de modifier le code actuel comme suit:

~~~cpp
Arcane::IItemFamily* cell_family = ...;
Arcane::ItemInternalArrayView cells = cell_family->itemsInternal();
Arcane::Int32 my_local_id = ...;
Arcane::Cell my_cell = cells[my_local_id];
~~~

Ce code est à remplacer par cela:

~~~cpp
Arcane::IItemFamily* cell_family = ...;
Arcane::CellInfoListView cells(cell_family);
Arcane::Int32 my_local_id = ...;
Arcane::Cell my_cell = cells[my_local_id];
~~~

Par la suite, si l'instance unique de Arcane::ItemSharedInfo par famille
est créée en mémoire unifiée, il sera possible d'accéder aux
informations des entités sur accélérateur.

## Modifications dans la version 3.10

%Arcane utilise très souvent pour gérer les entités du maillage des
listes de Arcane::ItemLocalId qui peuvent être ramenées à des listes
de Arcane::Int32. C'est par exemple utilisé dans les cas suivants:

- Liste des entités d'un groupe (Arcane::ItemGroup) ou d'un vecteur
  d'entité (Arcane::ItemVector)
- Liste des entités connectées à une autre entité ((par exemple
  Arcane::Cell::nodes()))

Pour ces deux cas la structure interne est gérée de la même manière
(via une instance de Arcane::ItemVectorView) et %Arcane maintient en
interne des objets de type Arcane::Int32ConstArrayView qui peuvent
être accédés directement par le développeur (par exemple via
Arcane::ItemVectorView::localIds()).

Pour gérer plus efficacement les connectivités, notamment dans le cas
cartésien, et réduire l'empreinte mémoire, il est nécessaire de
pouvoir faire évoluter la manière dont sont conservées ces listes de
localId(). Afin de pouvoir procéder à ces évolutions, il est
nécessaire de modifier deux choses dans %Arcane:

1. Séparer la gestion de la connectivité des entités de celle des
   liste d'entités d'un groupe.
2. Masquer la structure interne utilisée pour conserver ces listes de
   localId().

Cela implique de changer certains mécanismes pour accéder à ces
informations qui sont détaillés ci-dessous.

### Séparer la gestion de la connectivité des entités de celle des liste d'entités d'un groupe

Cela signifie que les méthodes d'accès aux connectivités et celles
d'accès aux entités d'un groupe ne retournent pas le même type
d'objet. Dans la version 3.9 de %Arcane, les méthodes d'accès aux
connectivités ont donc été modifées et retournent maintenant une
instance de Arcane::ItemConnectedListViewT au lieu d'une instance de
Arcane::ItemVectorViewT.

Il existe actuellement un opérateur de conversion entre
Arcane::ItemConnectedListViewT et Arcane::ItemVectorViewT pour rendre
le code compatible avec l'existant.

Cela impacte aussi les macros telles que ENUMERATE_(),
ENUMERATE_CELL() ou ENUMERATE_NODE() qui sont maintenant réservées aux
itérations sur les Arcane::ItemGroup ou
Arcane::ItemVector. Actuellement il y a plusieurs manières pour itérer
sur les entités d'une autre connectivité. Par exemple:

~~~cpp
Arcane::CellGroup cell_group = ...;
ENUMERATE_(Cell,icell,cell_group){
  Arcane::Cell cell = *icell;
  // (1) Itération avec ItemEnumerator
  for( Arcane::NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode ){
    Arcane::Node node = *inode;
    info() << "Node uid=" << node.uniqueId();
  }
  // (2) Itération avec ENUMERATE_
  ENUMERATE_(Node,inode,cell.nodes()){
    Arcane::Node node = *inode;
    info() << "Node uid=" << node.uniqueId();
  }
  // (3) Itération avec 'for-range'
  for( Arcane::Node node : cell.nodes()){
    info() << "Node uid=" << node.uniqueId();
  }
}
~~~

Le mécanisme (3) est à privilégier. A terme, le mécanisme (1) va
disparaitre car le type Arcane::ItemEnumerator sera réservé aux
itérations sur les groupes. Le mécanisme (2) pourrait continuer à être
disponible mais sera moins performant que le mécanisme (3).

Afin d'éviter aussi tout risque d'incompatibilité dans le futur, il
est préférable de ne pas utiliser directement les types des itérateurs
retournés mais d'utiliser le mot clé `auto` à la place.

### Masquer la structure interne gérant les listes de localId()

Pour masquer ces structures, les classes de %Arcane qui gèrent les
listes d'entités ne retourneront plus de types tels que
Arcane::Int32ConstArrayView. Par exemple les méthodes telles que
Arcane::ItemVectorView::localIds() ou
Arcane::ItemIndexArrayView::localIds() vont disparaitre. Pour être
compatible avec l'existant, les méthodes
Arcane::ItemVectorView::fillLocalIds() et
Arcane::ItemIndexArrayView::fillLocalIds() ont été ajoutées pour
permettre de remplir un tableau avec la liste des localId().

