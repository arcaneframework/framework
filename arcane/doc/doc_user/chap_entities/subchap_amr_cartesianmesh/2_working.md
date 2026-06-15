# Functionality {#arcanedoc_entities_amr_cartesianmesh_working}

[TOC]

This page explains how to use AMR with a Cartesian mesh.

A new structure allows access to AMR-related methods for Cartesian meshes: the
class `Arcane::CartesianMeshAMRMng`.

Its construction is simple:

```cpp
using namespace Arcane;

ICartesianMesh* cartesian_mesh = ICartesianMesh::getReference(mesh());
CartesianMeshAMRMng amr_mng(cartesian_mesh);
```

It is possible to choose the number of overlap cell layers using the method
`Arcane::CartesianMeshAMRMng::setOverlapLayerSizeTopLevel(Int32 new_size)`. By
default, there are no overlap cells for the highest level.

## Historical Adaptation API {#arcanedoc_entities_amr_cartesianmesh_working_histo}

\note API usable with both AMR types.

To refine or coarsen a zone of the mesh, there are two methods:

- `Arcane::CartesianMeshAMRMng::refineZone(const AMRZonePosition& position)`
- `Arcane::CartesianMeshAMRMng::coarseZone(const AMRZonePosition& position)`

The class `AMRZonePosition` defines a zone in the mesh. All active cells whose
barycenters are in this zone belong to this zone (active cell = having no
children).

For AMR type 3, it is necessary to have only active cells of the same level in
the zone.

### Refinement {#arcanedoc_entities_amr_cartesianmesh_working_histo_refine}

Once the `refineZone()` method is called, all cells in the zone will have child
cells. These child cells will be gathered into a patch.

AMR type 3 will create the necessary overlap cells and update the number of
overlap cell layers of the other patches.

If necessary, it is possible to call the method
`Arcane::CartesianMeshAMRMng::mergePatches()` to merge patches that can be
merged (if merging two patches creates a regular patch) (this is a simple merge:
no cell creation/deletion).

### Coarsening {#arcanedoc_entities_amr_cartesianmesh_working_histo_coarsen}

The `coarseZone()` method will delete the active cells in the zone.

For AMR type 1, the patches may then become irregular.

For AMR type 3, the modified patches will be subdivided so that they remain
regular. To avoid having too many patches, it is possible to call
`Arcane::CartesianMeshAMRMng::mergePatches()` immediately afterward.

## New Adaptation API {#arcanedoc_entities_amr_cartesianmesh_working_new}

\note AMR type 3 only.

(A complete and commented example is available here:
`arcane/src/arcane/tests/cartesianmesh/DynamicCircleAMRModule.cc`)

Compared to the other API, here, the user simply designates cells to refine in
the mesh. They do not have to construct one or more patches "by hand."

%Arcane will take care of creating one or more regular patches including at
least the cells marked by the user.
Other cells around these marked cells can therefore be refined in order to
create regular patches.

For the creation of these patches, there are two rules:

- the patches must have the highest possible efficiency (
  `|marked cells of the patch| / |cells of the patch|`),
- the number of patches must be as small as possible.

These two rules oppose each other.
To achieve maximum efficiency, one can create one patch per cell.
To have the fewest possible patches, it is enough to create a patch with all the
cells of the level.

Mesh adaptation is done in three phases.

### Adaptation Initialization {#arcanedoc_entities_amr_cartesianmesh_working_new_init}

Method
`Arcane::CartesianMeshAMRMng::beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first)`

```cpp
amr_mng.beginAdaptMesh(2, 0);
```
First, we initialize the adaptation by providing the maximum number of levels we
will need. This maximum allows calculating the number of overlap cell layers for
each level. If this number of levels is not reached, the number of layers will
have to be adjusted during the third phase (a few extra calculations).

The second argument is the level from which we start the adaptation.

If, during a previous iteration, we created a level that we wish to keep, we can
choose it here. The patches of this level will not be deleted, as will the
patches of the lower levels. The patches of the higher levels will be deleted to
be recreated in the second phase.

It is important to note that it is the patches that are deleted in this first
phase, not the cells of these patches. The cells (and the various items around
them), if they are no longer in any patch at the end of the second phase, will
be deleted in the third phase.

The consequence is that if a cell saw its patch deleted, but found a patch
during the second phase, the variables associated with it will not be reset.

Finally, it must be noted that an "InPatch" cell can become an "Overlap" cell,
and vice versa.

### Level-by-Level Adaptation {#arcanedoc_entities_amr_cartesianmesh_working_new_adapt}

Method
`Arcane::CartesianMeshAMRMng::adaptLevel(Int32 level_to_adapt, bool do_fatal_if_useless = false)`

```cpp
amr_mng.adaptLevel(level_to_adapt, true);
```
Second phase. Before calling this method, the cells of the patches at level
`level_to_adapt` that must be refined must have the `II_Refine` flag.

Example:

```cpp
CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
  if (m_amr[icell]) {
    icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
  }
}
```

The first argument is the level to adapt. Adaptation is done level by level, one
by one, from the lowest to the highest. It is possible to "restart" the
adaptation by calling this method with a level to adapt lower than the previous
call. In this case, the patches of levels higher than `level_to_adapt` will be
deleted (as during the first phase).

The second argument allows the program to crash if the call is useless (i.e., if
there are no `II_Refine` cells or if `level_to_adapt` is greater than the
previous call +1 (which implies there are no `II_Refine` cells)).

Once this method is called, the created patches are usable normally (their
directions are calculated; there is no need to call `computeDirections()`).

\note Nevertheless, using the `Arcane::CartesianConnectivity` connectivities
requires (for now) a call to `computeDirections()`.

### Adaptation End {#arcanedoc_entities_amr_cartesianmesh_working_new_end}

Method `Arcane::CartesianMeshAMRMng::endAdaptMesh()`

```cpp
amr_mng.endAdaptMesh();
```
Finally, the last phase.

This phase will first adjust the number of overlap cell layers of each patch in
the case where the maximum number of levels given during the first phase is not
reached.

Then, it will delete all cells that do not have the `II_InPatch` flag or the
`II_Overlap` flag.

A call to the method `Arcane::CartesianMeshAMRMng::clearRefineRelatedFlags()`
will also be made.




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_amr_cartesianmesh_notions
</span>
<span class="next_section_button">
\ref arcanedoc_entities_connectivity_internal
</span>
</div>
