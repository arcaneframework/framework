# Notions {#arcanedoc_entities_amr_cartesianmesh_notions}

[TOC]

Cette page explique quelques notions pour comprendre le fonctionnement de l'AMR dans %Arcane.

## Patch {#arcanedoc_entities_amr_cartesianmesh_notions_patch}

Classe `Arcane::CartesianPatch`.

Pour l'AMR type 1, un patch est un ensemble de mailles. Ces mailles ne forment pas forcément un ensemble contigü.
Ces mailles sont regroupées dans un groupe de mailles accessible via la méthode `Arcane::CartesianPatch::cells()`.

\image html amr_1.webp


Pour l'AMR type 3, un patch est un ensemble de mailles d'un même niveau et englobé au sein d'une boite englobante (patch
régulier). Cette boite englobante est décrite par les coordonnées topologiques de deux mailles dans la grille
cartésienne : `min` et `max`.

\remark `min` et `max` auront les mêmes valeurs en multi sous-domaine ou en mono sous-domaine.

\image html amr_2.webp

Cette boite englobante est décrite par la classe `Arcane::AMRPatchPosition`. Chaque patch contient une instance de cette
classe, accessible via la méthode `Arcane::CartesianPatch::position()`.

\note Une maille ne peut être que dans une seule boite englobante (pour un niveau donné).

Trois groupes de mailles sont accessibles pour chaque patch :
- le groupe de toutes les mailles du patch : `Arcane::CartesianPatch::cells()`,
- le groupe des mailles de recouvrement : `Arcane::CartesianPatch::overlapCells()` (ayant le flag `II_Overlap`),
- le groupe des mailles du patch (non de recouvrement) : `Arcane::CartesianPatch::inPatchCells()`
  (ayant le flag `II_InPatch`).


## Mailles de recouvrement {#arcanedoc_entities_amr_cartesianmesh_notions_overlap}

\note Pour l'AMR type 1, il n'y a pas de mailles de recouvrement.

Les mailles de recouvrement désignent les mailles autour des patchs (autour des boites englobantes).

\image html amr_3.webp

(en pointillé, on a les mailles/faces/noeuds de recouvrements / 2 couches pour le niveau 1)

Ces mailles permettent deux choses. D'abord, elles permettent d'obtenir les valeurs autour des items du patch (en trait
plein sur l'image) (comme les mailles fantômes pour le calcul à la frontière des sous-domaines).

\image html amr_4.webp

(on peut voir des mailles de recouvrement qui recouvrent des mailles d'autres patchs (`II_Overlap && II_InPatch`))

Ensuite, elles permettent d'éviter d'avoir plus d'un niveau de différence entre deux mailles.
En effet, il n'est pas possible de raffiner une maille de recouvrement pure (`II_Overlap && ! II_InPatch`).

\image html amr_5.webp

(2 couches pour le niveau 1 / 0 couche pour le niveau 2)

Il est possible de modifier le nombre de couches de mailles de recouvrement du niveau le plus haut via la méthode
`Arcane::CartesianMeshAMRMng::setOverlapLayerSizeTopLevel(Int32 new_size)`.
Le nombre de couches des autres niveaux sera calculé automatiquement.

Il est aussi possible de désactiver la création de ces couches avec la méthode
`Arcane::CartesianMeshAMRMng::disableOverlapLayer()`. Dans ce cas, il pourra y avoir plus d'un niveau de différence
entre niveaux.

## Directions {#arcanedoc_entities_amr_cartesianmesh_notions_directions}

(Lire la page \ref arcanedoc_entities_cartesianmesh_direction avant de continuer)

Chaque patch (pour les deux types d'AMR) possède ses propres directions, pour chaque item.

Ces directions sont accessibles via les patchs (`Arcane::CartesianPatch`). Le fonctionnement est le même
qu'avec le maillage sans AMR. 

Deux nouvelles méthodes sont disponibles pour accéder aux groupes d'items `InPatch` et `Overlap` :

- `Arcane::CellDirectionMng::inPatchCells()` et `Arcane::CellDirectionMng::overlapCells()`,
- `Arcane::FaceDirectionMng::inPatchFaces()` et `Arcane::FaceDirectionMng::overlapFaces()`,
- `Arcane::NodeDirectionMng::inPatchNodes()` et `Arcane::NodeDirectionMng::overlapNodes()`.

Exemple :

```cpp
using namespace Arcane;

ICartesianMesh* cartesian_mesh = ICartesianMesh::getReference(mesh());
CartesianMeshAMRMng amr_mng(cartesian_mesh);
CartesianPatch patch = amr_mng.amrPatch(1);

FaceDirectionMng face_dm(patch.faceDirection(MD_DirX));
ENUMERATE_(Face, iface, face_dm.inPatchFaces()) {
  Face face = *iface;
  DirFace dir_face(face_dm[iface]);
  Cell prev_cell = dir_face.previousCell(); // Maille avant la face
  Cell next_cell = dir_face.nextCell(); // Maille après la face
}
```
Dans ce bout de code, avec au moins une couche de mailles de recouvrement, on est sûr que `dir_face.previousCell()` et
`dir_face.nextCell()` ne sont pas nulles (sauf au bord du sous-domaine).

\image html amr_6.webp




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_amr_cartesianmesh
</span>
<span class="next_section_button">
\ref arcanedoc_entities_amr_cartesianmesh_working
</span>
</div>
