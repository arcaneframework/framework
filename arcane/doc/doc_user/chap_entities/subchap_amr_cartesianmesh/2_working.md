# Fonctionnement {#arcanedoc_entities_amr_cartesianmesh_working}

[TOC]

Cette page explique comment utiliser l'AMR avec un maillage cartésien.

Une nouvelle structure permet d'accéder aux méthodes liées à l'AMR pour les maillages cartésiens : la classe
`Arcane::CartesianMeshAMRMng`.

Sa construction est simple :

```cpp
using namespace Arcane;

ICartesianMesh* cartesian_mesh = ICartesianMesh::getReference(mesh());
CartesianMeshAMRMng amr_mng(cartesian_mesh);
```

Il est possible de choisir le nombre de couches de mailles de recouvrement avec la méthode
`Arcane::CartesianMeshAMRMng::setOverlapLayerSizeTopLevel(Int32 new_size)`. Par défaut, il n'y a pas de mailles de
recouvrement pour le niveau le plus haut.

## API d'adaptation historique {#arcanedoc_entities_amr_cartesianmesh_working_histo}

\note API utilisable avec les deux types d'AMR.

Pour raffiner ou dé-raffiner une zone du maillage, on a deux méthodes :

- `Arcane::CartesianMeshAMRMng::refineZone(const AMRZonePosition& position)`
- `Arcane::CartesianMeshAMRMng::coarseZone(const AMRZonePosition& position)`

La classe `AMRZonePosition` définit une zone dans le maillage. Toutes les mailles actives ayant leurs barycentres dans
cette zone appartiennent à cette zone (maille active = n'ayant pas d'enfants).

Pour l'AMR type 3, il est nécessaire d'avoir uniquement des mailles actives du même niveau dans la zone.

### Raffinement {#arcanedoc_entities_amr_cartesianmesh_working_histo_refine}

Une fois la méthode `refineZone()` appelée, toutes les mailles de la zone auront des mailles filles. Ces mailles
filles seront réunies dans un patch.

L'AMR type 3 créera les mailles de recouvrement nécessaires et mettra à jour le nombre de couches de mailles de
recouvrement des autres patchs.

Si nécessaire, il est possible d'appeler la méthode `Arcane::CartesianMeshAMRMng::mergePatches()` afin de fusionner les
patchs pouvant l'être (si la fusion de deux patchs crée un patch régulier) (c'est une fusion simple : pas de
création/suppression de mailles).

### Dé-raffinement {#arcanedoc_entities_amr_cartesianmesh_working_histo_coarsen}

La méthode `coarseZone()` supprimera les mailles actives de la zone.

Pour l'AMR type 1, les patchs peuvent alors devenir irréguliers.

Pour l'AMR type 3, il y aura découpage des patchs modifiés afin qu'ils restent réguliers. Pour éviter d'avoir trop de
patchs, il est possible d'appeler `Arcane::CartesianMeshAMRMng::mergePatches()` juste après.

## Nouvelle API d'adaptation {#arcanedoc_entities_amr_cartesianmesh_working_new}

\note AMR type 3 seulement.

(Un exemple complet et commenté est disponible ici : `arcane/src/arcane/tests/cartesianmesh/DynamicCircleAMRModule.cc`)

Par rapport à l'autre API, ici, l'utilisateur désigne simplement des mailles à raffiner dans le maillage. Il n'a pas à
construire un ou plusieurs patchs "à la main".

%Arcane s'occupera de créer un ou plusieurs patchs réguliers incluant au moins les mailles marquées par l'utilisateur.
D'autres mailles autour de ces mailles marquées peuvent donc être raffiné afin de créer des patchs réguliers.

Pour la création de ces patchs, il y a deux règles :

- les patchs doivent avoir une efficacité la plus grande possible (`|mailles marquées du patch| / |mailles du patch|`),
- le nombre de patchs doit être le plus petit possible.

Ces deux règles s'opposent.
Pour avoir une efficacité maximale, on peut faire un patch par maille.
Pour avoir le moins de patchs possibles, il suffit de faire un patch avec toutes les mailles du niveau.

L'adaptation du maillage se fait en trois phases.

### Initialisation de l'adaptation {#arcanedoc_entities_amr_cartesianmesh_working_new_init}

Méthode `Arcane::CartesianMeshAMRMng::beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first)`

```cpp
amr_mng.beginAdaptMesh(2, 0);
```
D'abord, on initialise l'adaptation en donnant le nombre maximum de niveaux dont on aura besoin. Ce maximum permet de
calculer le nombre de couches de mailles de recouvrement pour chaque niveau. Si ce nombre de niveaux n'est pas atteint,
le nombre de couches devra être ajusté lors de la troisième phase (quelques calculs en plus).

Le deuxième argument est le niveau à partir duquel on commence l'adaptation.

Si, lors d'une précédente itération, on a créé un niveau que l'on souhaite conserver, on peut le choisir ici. Les patchs
de ce niveau ne seront pas effacés, ainsi que les patchs des niveaux inférieurs. Les patchs des niveaux supérieurs
seront effacés pour être recréés dans la seconde phase.

Il est important de noter que ce sont les patchs qui sont supprimés dans cette première phase, pas les mailles de ces
patchs. Les mailles (et les différents items autours), si elles ne sont plus dans aucun patch à l'issue de la seconde
phase, seront supprimées dans la troisième phase.

La conséquence est que, si une maille à vu son patch être supprimé, mais a retrouvé un patch lors de la seconde phase,
les variables qui lui sont associées ne seront pas réinitialisées.

Enfin, il faut noter qu'une maille "InPatch" peut devenir une maille "Overlap", et inversement.

### Adaptation niveau par niveau {#arcanedoc_entities_amr_cartesianmesh_working_new_adapt}

Méthode `Arcane::CartesianMeshAMRMng::adaptLevel(Int32 level_to_adapt, bool do_fatal_if_useless = false)`

```cpp
amr_mng.adaptLevel(level_to_adapt, true);
```
Deuxième phase. Avant d'appeler cette méthode, les mailles des patchs du niveau `level_to_adapt` qui doivent être
raffinées doivent avoir le flag `II_Refine`.

Exemple :

```cpp
CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
  if (m_amr[icell]) {
    icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
  }
}
```

Le premier argument est le niveau à adapter. L'adaptation se fait niveau par niveau, un par un, du plus bas vers le plus
haut. Il est possible de "recommencer" l'adaptation en appelant cette méthode avec un niveau à adapter inférieur à
l'appel précédent. Dans ce cas, les patchs de niveau supérieur à `level_to_adapt` seront supprimés (comme lors de la
première phase).

Le second argument permet de faire planter le programme si l'appel est inutile (c'est-à-dire s'il n'y a pas de mailles
`II_Refine` ou si `level_to_adapt` est supérieur au précedent appel +1 (ce qui implique qu'il n'y a pas de mailles
`II_Refine`)).

Une fois cette méthode appelée, les patchs créés sont utilisables normalement (leurs directions sont calculées ; pas
besoin d'appeler `computeDirections()`).

\note Néanmoins, l'utilisation des connectivités `Arcane::CartesianConnectivity` nécessite (pour l'instant) un appel à
`computeDirections()`.

### Fin de l'adaptation {#arcanedoc_entities_amr_cartesianmesh_working_new_end}

Méthode `Arcane::CartesianMeshAMRMng::endAdaptMesh()`

```cpp
amr_mng.endAdaptMesh();
```
Enfin, la dernière phase.

Cette phase va d'abord ajuster le nombre de couches de mailles de recouvrement de chaque patch dans le cas où le nombre
de niveaux maximum donné lors de la première phase n'est pas atteint.

Puis, elle va supprimer toutes les mailles qui n'ont ni le flag `II_InPatch`, ni le flag `II_Overlap`.

Un appel à la méthode `Arcane::CartesianMeshAMRMng::clearRefineRelatedFlags()` sera aussi effectué.




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_amr_cartesianmesh_notions
</span>
<span class="next_section_button">
\ref arcanedoc_entities_connectivity_internal
</span>
</div>
