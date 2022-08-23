Modifications dans la gestions des entités avec la version 3.7.
=======================

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

~~~{.cpp}
Arcane::IItemFamily* cell_family = ...;
Arcane::ItemInternalArrayView cells = cell_family->itemsInternal();
Arcane::Int32 my_local_id = ...;
Arcane::Cell my_cell = cells[my_local_id];
~~~

Ce code est à remplacer par cela:

~~~{.cpp}
Arcane::IItemFamily* cell_family = ...;
Arcane::CellInfoListView cells(cell_family);
Arcane::Int32 my_local_id = ...;
Arcane::Cell my_cell = cells[my_local_id];
~~~

Par la suite, si l'instance unique de Arcane::ItemSharedInfo par famille
est créée en mémoire unifiée, il sera possible d'accéder aux
informations des entités sur accélérateur.

