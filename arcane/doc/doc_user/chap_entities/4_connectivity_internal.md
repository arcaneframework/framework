# Gestion des connectivités des entités {#arcanedoc_entities_connectivity_internal}

[TOC]

Cette page regroupe les informations sur les développements
effectués dans %Arcane pour la gestion des nouvelles
connectivités. Il se base sur la version CEA de février 2017, ce qui
correspond aux versions de %Arcane 2.5.0 et ultérieures.

Afin de répondre à de nouveaux besoin, le mécanisme de gestion des
connectivités des entités de %Arcane a évolué à partir de 2017.

Le mécanisme historique avait pour but premier d'économiser la mémoire
et rangeait toutes les connectivités d'une entité consécutivement en
mémoire. Cela présente cependant deux inconvénients :
- ce mécanisme n'est pas facilement extensible si on souhaite ajouter de
  nouvelles connectivités ou si on souhaite ne pas utiliser certaines
  connectivités. Au départ, seuls les noeuds, faces et mailles étaient
  gérés. Aujourd'hui il y a les noeuds duaux, les liens, les degrés de
  liberté, l'AMR et toutes ces connectivités alourdissent la gestion.
- il est plus difficile de profiter des effets de cache mémoire
  lorsqu'on parcourt qu'un seul type de connectivité.

Le nouveau mécanisme permet de séparer complètement chaque type de
connectivités et éventuellement de spécialiser un type de
connectivité en fonction de certains besoins (par exemple suivant le
type de maillage).

Il permet de résoudre les deux inconvénients
précédents avec en contre-partie pour les maillages non structurés
une augmentation de la mémoire utilisée. Avec l'ancien mécanisme,
pour chaque entité 1 indice (de type Arcane::Int32) suffisait pour accéder aux infos
de connectivité alors qu'avec le nouveau il faut 2 indices (position +
nombre de connectivités) par connectivité. Par exemple dans le cas
des entités classiques de maillage (noeud, arête, face ou maille),
il faut donc 8 indices au lieu de 1.

Le mécanisme historique permet d'accéder aux informations de
connectivité directement via l'entité. Par exemple, pour accéder
au 4-ème noeud d'une maille :
```cpp
Arcane::Cell cell;
Arcane::Node node = cell.node(3);
```

\`A terme, l'accès aux connectivités pourrait être disponible sous une autre forme
mais en attendant il faut pouvoir continuer à utiliser ce mécanisme
sous peine de rendre tous les codes actuels incompatibles.

Pour effectuer la transition entre l'ancienne et la nouvelle gestion
des connectivités, des mécanismes de compatibilité ont été mis en
place. L'objectif de ces mécanismes est de garantir la compatibilité
au niveau des sources des codes utilisant %Arcane: ces codes doivent
pouvoir compiler sans modification avec les versions d'%Arcane
intégrant les nouvelles connectivités.
  
Pour garantir cette compatibilité, le mécanisme d'accès pour les
méthodes telles que Arcane::Cell::node() est modifié et
utilise maintenant un objet de type Arcane::ItemInternalConnectivityList. La
classe Arcane::ItemInternal contient un champ Arcane::ItemInternal::m_connectivity
est qui un pointeur sur un Arcane::ItemInternalConnectivityList. Toutes les
entités d'une même famille pointent sur la même valeur qui est
Arcane::ItemFamily::m_item_connectivity_list.

\note Le fait de mettre ce pointeur dans chaque Arcane::ItemInternal permet
d'éviter une indirection lorsqu'on accède aux connectivités mais il
est aussi possible de le mettre dans le Arcane::ItemSharedInfo car il est
commun à toutes les entités d'une famille. Cela permet d'utiliser
moins de mémoire pour Arcane::ItemInternal (16 octets au lieu de 24) au pris
d'une indirection supplémentaire. Dans mes tests au CEA, je n'ai pas
eu de différences de performances entre les deux mécanismes.

Afin de ne pas modifier l'API existante, les méthodes de ItemInternal
permettant d'accéder à la connectivité n'ont pas été modifiées et de
nouvelles ont été ajoutées. Elles utilisent le suffixe V2. Par
exemple Arcane::ItemInternal::nodesV2() au lieu de Arcane::ItemInternal::nodes().

La macro **ARCANE_USE_LEGACY_ITEMINTERNAL_CONNECTIVITY** permet de
choisir à la compilation si les acceseseurs via Arcane::Item, Arcane::Edge, Arcane::Face,
Arcane::Cell, ... utilisent les anciens ou nouveaux mécanismes. Si cette
macro est définie alors :
```cpp
// version historique si macro définie
NodeVectorView Cell::node() { return m_internal->nodes(); }
// nouvelle version si macro non définie
NodeVectorView Cell::node() { return m_internal->nodesV2(); }
```

Cette macro est définie uniquement si on compile %Arcane avec dans
le configure l'option **--with-legacy-connectivity**. Si cette option
est active, il est impossible d'accéder aux nouvelles connectivités
via Item. Le seul intérêt de cette option est de vérifier si le
nouveau mécanisme ne contient pas de bugs et de valider les
nouvelles versions de %Arcane sur d'anciens codes utilisateur. On
supposera par la suite que cette option n'est pas utilisée et donc
qu'on accède aux connectivités via les méthodes V2. Pour cela, il
faut utiliser l'option de configuration
**--without-legacy-connectiviy** dans le configure.

Lorsqu'on utilise les méthodes V2, les accès de chaque connectivité
se font via la classe Arcane::ItemInternalConnectivityList. Pour chaque
connectivité, il y a trois tableaux :
- nb_item: nombre d'entités connectée
- list: tableaux des localId() des entités connectées.
- index: indice dans \a list de la première entité connectée.

La liste des entités connectées à une entité est rangée
consécutivement en mémoire et donc l'indice dans le tableau de la
première permet de récupérer les autres.
\a nb_item et \a index sont indexés par le localId() de l'entité
dont on souhaite avoir les connectivités. Par exemple :

```cpp
using namespace Arcane;
Item my_item = ...;
Int32 lid = my_item.localId();
Int32ConstArrayView nb_items = ...;
Int32ConstArrayView list = ...;
Int32ConstArrayView index = ...;
// Nombre d'entités connectés à my_item.
Int32 n = nb_items[lid];
// localId() de la première entité connectées à my_item.
Int32 c0 = list[ index[lid] ];
```

Toutes les connectivités des entités classiques (Arcane::Node, Arcane::Edge, Arcane::Face et
Arcane::Cell) utilisent maintenant ce type d'accès. Il est possible de choisir
à l'exécution si les accès se font via les connectivités historiques
ou les nouvelles connectitivités. Cela se fait via la variable
d'environnement **ARCANE_CONNECTIVITY_POLICY** qui permet d'associer une
des valeurs énumérées #Arcane::InternalConnectivityPolicy. Cette association
se fait dans le constructeur de DynamicMesh. La méthode
Arcane::IMesh::_connectivityPolicy() permet de récupérer la politique
choisie. Il y a actuellement 4 valeurs possibles :
- Arcane::InternalConnectivityPolicy::Legacy: indique seules les
  connectivités historiques sont allouées et donc que
  ItemInternal::m_connectivity n'est pas utilisé. Il n'est donc pas
  possible d'accéder aux nouveaux mécanismes de connectivités via
  ce mode. Ce mode est celui qui correspond le plus aux anciennes
  versions de Arcane, notamment au niveau de l'usage mémoire.
- Arcane::InternalConnectivityPolicy::LegacyAndAllocAccessor: indique que seules les
  connectivités historiques sont allouées et que
  Arcane::ItemInternal::m_connectivity utilise les connectivités définies dans
  Arcane::ItemFamily::m_items_data. Le booléen
  Arcane::ItemFamily::m_use_legacy_connectivity_policy vaut \a true dans
  ce cas. Ce mode alloue les acceseurs des connectivités via
  ItemInternal::m_connectivity et donc utilise plus de mémoire que le
  mode InternalConnectivityPolicy::Legacy.
- Arcane::InternalConnectivityPolicy::LegacyAndNew: est identique à la
  valeur InternalConnectivityPolicy::LegacyAndAllocAccessor mais en plus les nouvelles
  connectités sont allouées. Elles ne sont cependant pas utilisées par
  les classes Item et Internal. En mode check, on vérifie à chaque
  modification de maillage que les valeurs des anciennes et nouvelles
  connectivités sont les mêmes. Ce mode permet donc de valider les
  nouveaux mécanismes. Pour ce mode
  Arcane::ItemFamily::m_use_legacy_connectivity_policy vaut aussi \a true.
- Arcane::InternalConnectivityPolicy::NewAndLegacy: indique qu'on alloue les
  anciennes et nouvelles connectivités mais que les accès via Item et
  ItemInternal se font avec ces nouvelles connectivités. Ce mode est
  donc proche du futur mode de fonctionnement.

\`A terme il y a aura une 5-ème valeur qui correspond au mode définitif
ou seules les nouvelles connectivités sont allouées. Il sera mis en
place lorsque tous les codes utilisant %Arcane auront été validés
avec les nouvelles connectivités.

Suivant comme %Arcane est configuré, certaines valeurs ne sont pas
possibles. Si la configuration est faite avec
**--with-legacy-connectivity**, alors le mode
Arcane::InternalConnectivityPolicy::NewAndLegacy n'est pas possible. Si
%Arcane est configuré avec **--without-legacy-connectivity**, alors
le mode Arcane::InternalConnectivityPolicy::Legacy n'est pas possible.




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_amr_cartesianmesh
</span>
<span class="next_section_button">
\ref arcanedoc_entities_itemtype
</span>
</div>
