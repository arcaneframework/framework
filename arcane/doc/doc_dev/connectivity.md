# Gestion des connectivités à la demande {#arcanedoc_connectivity}

[TOC]

## État actuel {#arcanedoc_connectivity_current}

Actuellement, l'interface IItemConnectivity permet de gèrer de nouvelles
connectivités mais elle n'est pas prévues pour une mise à jour
incrémentale des éléments. Par incrémental, on entend que les
connectivités doivent être mises à jour immédiatement après l'ajout
où la supression d'une entité.

Comme convenu lors d'une réunion avec Stéphane, une nouvelle
interface IIncrementalItemConnectivity a été créée pour gérer ce type
de connectivité. Une implémentation générique est disponible et
s'apelle IncrementalItemConnectivity. Elle n'est pas optimisée mais
peut s'appliquer à n'importe quel type de connectivité

L'interface IIncrementalItemConnectivity possède plusieurs méthodes
réparties en trois catégories:

- les méthodes identiques à IItemConnectivity. Il s'agit de
IItemConnectivity::name(), IItemConnectivity::families(),
IItemConnectivity::sourceFamily() et
IItemConnectivity::targetFamily().

- les méthodes d'ajout/supression. Il s'agit de
IIncrementalItemConnectivity::addConnectedItem() et
IIncrementalItemConnectivity::removeConnectedItem(). Ces deux méthodes
permettent d'ajouter ou suprimer une entité.
\note Pour l'instant ces méthodes prennent en argument un
ItemInternal*. Il faudrait voir si ce ne serait pas mieux avec un
ItemLocalId.
- les méthodes de notification. Elles sont disponibles pour que les
familles associées à ces connectivités puissent notifier ces
dernières d'une modification interne. Il y a 4
méthodes. IIncrementalItemConnectivity::notifySourceFamilyLocalIdChanged() et
IIncrementalItemConnectivity::notifyTargetFamilyLocalIdChanged() sont appelés
lorsque la famille source ou cible est compactée. La méthode
IIncrementalItemConnectivity::notifySourceItemAdded() est appelé lorsqu'un
élément est ajouté à la famille source. Cela permet notamment de
redimensionner les tableaux internes. Enfin, la dernière méthode
IIncrementalItemConnectivity::notifyReadFromDump() est appelé après une
relecture suite à un retour-arrière où une reprise.

Ces méthodes de notification peuvent évoluer de plusieurs manières:
- on peut ajouter aux familles la notion d'évènement, avec un
évènement par type de notification et ensuite chaque famille
s'enregistre. Cela a l'avantage de ne pas nécessiter de méthode
spécifique dans l'interface IIncrementalItemConnectivity mais rend le
code moins lisible (car l'enregistrement des familles ne se voit pas
dans facilement dans le source) et ne permet pas de gérer facilement
l'ordre des appels entre différentes connectivités si besoin.
- on peut aussi rendre disponible ces notifications aux connectivités
qui implémentent IItemConnectivity. Dans ce cas une interface de base
commune avec IIncrementalItemConnectivity serait utile.

Pour gérer correctement les mises à jour suite au compactage, j'ai du
rendre privé à ItemFamily l'accès en écriture à
ItemFamily::m_infos. Du coup, au lieu d'appeler le compactage des
entités directement via DynamicMeshKindInfos::beginCompactItems() et
DynamicMeshKindInfos::finishCompactItems(), il faut appeler les
méthodes correspondantes de ItemFamily qui va faire la délégation et
notifier les connectivités incrémentales de ce changement.
\note C'est à cet endroit qu'on pourrait aussi notifier aux autres
connectités d'un éventuel compactage.

Ces nouvelles fonctionnalités ne sont pour l'instant implémentées que
pour la connectivité noeud->face de NodeFamily. Cette connectivité se
fait en double avec la connectivité classique. En mode check, on
vérifie après chaque changement dans la famille que la nouvelle
connectivité et l'ancienne (qui sert de référence) sont les
mêmes.
\warning Par contre, actuellement la nouvelle connectivité n'est jamais
utilisée directement. L'accès via ItemInternal se fait toujours avec
l'ancien mécanisme.
\note Il n'y a actuellement qu'une seule opération qui n'est pas
implémenté avec les nouvelles fonctionnalités, c'est le tri par
uniqueId() croissant fait dans la méthode
DynamicMesh::_sortInternalReferences().

Pour cette connectivité noeud->face, j'ai aussi implémenté la
connectivité actuelle via l'interface
IIncrementalItemConnectivity. La classe qui gère cela est
NodeFaceCompactIncrementalItemConnectivity. Du coup, le même
mécanisme est utilisé pour l'ancienne et la nouvelle connectivité et
il est donc assez général.

Pour activer la nouvelle connectivité, il faut positionner la
variable d'environnement ARCANE_CONNECTIVITY_POLICY à 1. A terme
évidemment il faudra faire autrement.

Actuellement, tous les cas de la base de test fonctionnent avec cette
nouvelle connectivité. J'ai aussi testé (sommairement) sur la base
d'intégration de nos codes sous Arcane et je n'ai pas eu de problèmes.

## Prochaines phases {#arcanedoc_connectivity_next_phases}

Si la preuve de concept est ok, les phases suivantes seront (plus ou
moins dans cet ordre):
- modifier ItemInternal pour utiliser la nouvelle connectivité si elle
est définie.
- utiliser l'interface IIncrementalItemConnectivity pour gérer toutes
les connectivités, même les anciennes. Cette phase peut se faire en
plusieurs sous-phases suivant les connectivités. Les connectivités
'classiques' noeuds, arêtes, faces et mailles d'abord, les
connectivités plus compliquées (notamment AMR) ensuite.
- optimiser IncrementalItemConnectivity notamment en gérant la
pré-allocation pour ne pas réallouer à chaque fois. Il faudra aussi
gérer le compactage.
- pouvoir activer par configuration l'ancienne où la nouvelle
connectivité.
