# Notions {#arcanedoc_entities_amr_cartesianmesh_notions}

[TOC]

This page explains some concepts to understand how AMR works in %Arcane.

## Patch {#arcanedoc_entities_amr_cartesianmesh_notions_patch}

Class `Arcane::CartesianPatch`.

For AMR type 1, a patch is a set of meshes. These meshes do not necessarily form
a contiguous set.
These meshes are grouped into a set of meshes accessible via the method
`Arcane::CartesianPatch::cells()`.

\image html amr_1.webp


For AMR type 3, a patch is a set of meshes of the same level and enclosed within
a bounding box (regular patch). This bounding box is described by the
topological coordinates of two meshes in the Cartesian grid: `min` and `max`.

\remark `min` and `max` will have the same values in a multi-subdomain or a
mono-subdomain.

\image html amr_2.webp

This bounding box is described by the class `Arcane::AMRPatchPosition`. Each
patch contains an instance of this class, accessible via the method
`Arcane::CartesianPatch::position()`.

\note A mesh can only be in one bounding box (for a given level).

Three sets of meshes are accessible for each patch:
- the group of all meshes in the patch: `Arcane::CartesianPatch::cells()`,
- the group of overlap meshes: `Arcane::CartesianPatch::overlapCells()` (having
  the `II_Overlap` flag),
- the group of patch meshes (non-overlapping):
  `Arcane::CartesianPatch::inPatchCells()` (having the `II_InPatch` flag).


## Overlap Meshes {#arcanedoc_entities_amr_cartesianmesh_notions_overlap}

\note For AMR type 1, there are no overlap meshes.

Overlap meshes refer to the meshes around the patches (around the bounding
boxes).

\image html amr_3.webp

(In dotted lines, we have the overlap meshes/faces/nodes / 2 layers for level 1)

These meshes allow two things. First, they allow obtaining values around the
patch items (in solid lines in the image) (like ghost meshes for calculating at
the subdomain boundary).

\image html amr_4.webp

(We can see overlap meshes covering meshes from other patches
(`II_Overlap && II_InPatch`))

Second, they prevent having more than one level of difference between two
meshes.
Indeed, it is not possible to refine a pure overlap mesh
(`II_Overlap && ! II_InPatch`).

\image html amr_5.webp

(2 layers for level 1 / 0 layers for level 2)

It is possible to modify the number of overlap mesh layers of the highest level
via the method
`Arcane::CartesianMeshAMRMng::setOverlapLayerSizeTopLevel(Int32 new_size)`.
The number of layers for other levels will be calculated automatically.

It is also possible to disable the creation of these layers with the method
`Arcane::CartesianMeshAMRMng::disableOverlapLayer()`. In this case, there may be
more than one level of difference between levels.

## Directions {#arcanedoc_entities_amr_cartesianmesh_notions_directions}

(Read the page \ref arcanedoc_entities_cartesianmesh_direction before
continuing)

Each patch (for both AMR types) has its own directions, for each item.

These directions are accessible via the patches (`Arcane::CartesianPatch`). The
operation is the same as with the mesh without AMR.

Two new methods are available to access the `InPatch` and `Overlap` item groups:

- `Arcane::CellDirectionMng::inPatchCells()` and
  `Arcane::CellDirectionMng::overlapCells()`,
- `Arcane::FaceDirectionMng::inPatchFaces()` and
  `Arcane::FaceDirectionMng::overlapFaces()`,
- `Arcane::NodeDirectionMng::inPatchNodes()` and
  `Arcane::NodeDirectionMng::overlapNodes()`.

Example:

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
In this piece of code, with at least one layer of overlap meshes, we are sure
that `dir_face.previousCell()` and
`dir_face.nextCell()` are not null (except at the subdomain boundary).

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
