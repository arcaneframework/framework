// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatchGroup.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Management of the Cartesian mesh patch group.                             */
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/CartesianPatchGroup.h"

#include "arcane/utils/FixedArray.h"
#include "arcane/utils/Vector3.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/Properties.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/AMRZonePosition.h"

#include "arcane/cartesianmesh/internal/CartesianMeshPatch.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"
#include "arcane/cartesianmesh/internal/AMRPatchPositionLevelGroup.h"
#include "arcane/cartesianmesh/internal/AMRPatchPositionSignature.h"
#include "arcane/cartesianmesh/internal/AMRPatchPositionSignatureCut.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshNumberingMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianPatchGroup::
CartesianPatchGroup(ICartesianMesh* cmesh)
: TraceAccessor(cmesh->traceMng())
, m_cmesh(cmesh)
, m_index_new_patches(1)
, m_size_of_overlap_layer_top_level(0)
, m_higher_level(0)
, m_target_nb_levels(0)
, m_latest_call_level(-2)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
build()
{
  m_properties = makeRef(new Properties(*(m_cmesh->mesh()->properties()), "CartesianPatchGroup"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
saveInfosInProperties()
{
  m_properties->set("Version", 1);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    UniqueArray<String> patch_group_names;
    for (Integer i = 1; i < m_amr_patches_pointer.size(); ++i) {
      patch_group_names.add(allCells(i).name());
    }
    m_properties->set("PatchGroupNames", patch_group_names);
  }

  else {
    // It is unnecessary to save the ground patch. It will be recalculated.
    UniqueArray<String> patch_group_names(m_amr_patches_pointer.size() - 1);
    UniqueArray<Int32> level(m_amr_patches_pointer.size() - 1);
    UniqueArray<Int32> overlap(m_amr_patches_pointer.size() - 1);
    UniqueArray<Int32> index(m_amr_patches_pointer.size() - 1);
    UniqueArray<CartCoord> min_point((m_amr_patches_pointer.size() - 1) * 3);
    UniqueArray<CartCoord> max_point((m_amr_patches_pointer.size() - 1) * 3);

    for (Integer patch = 1; patch < m_amr_patches_pointer.size(); ++patch) {
      const Integer pos_in_array = patch - 1;
      const AMRPatchPosition& position = m_amr_patches_pointer[patch]->_internalApi()->positionRef();
      level[pos_in_array] = position.level();
      overlap[pos_in_array] = position.overlapLayerSize();
      index[pos_in_array] = m_amr_patches_pointer[patch]->index();

      const Integer pos = pos_in_array * 3;
      min_point[pos + 0] = position.minPoint().x;
      min_point[pos + 1] = position.minPoint().y;
      min_point[pos + 2] = position.minPoint().z;
      max_point[pos + 0] = position.maxPoint().x;
      max_point[pos + 1] = position.maxPoint().y;
      max_point[pos + 2] = position.maxPoint().z;

      patch_group_names[pos_in_array] = allCells(patch).name();
    }
    m_properties->set("LevelPatches", level);
    m_properties->set("OverlapSizePatches", overlap);
    m_properties->set("IndexPatches", index);
    m_properties->set("MinPointPatches", min_point);
    m_properties->set("MaxPointPatches", max_point);

    // TODO: Find another way to handle this.
    //        In the case of a resumed protection, the m_available_index
    //        array cannot be correctly recalculated because of elements after
    //        the "max index" of the "active indices". These "extra" elements cannot be found without more information.
    m_properties->set("PatchGroupNamesAvailable", m_available_group_index);
    m_properties->set("PatchGroupNames", patch_group_names);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
recreateFromDump()
{
  Trace::Setter mci(traceMng(), "CartesianPatchGroup");

  // Save the version number to ensure it is OK upon restart
  Int32 v = m_properties->getInt32("Version");
  if (v != 1) {
    ARCANE_FATAL("Bad serializer version: trying to read from incompatible checkpoint v={0} expected={1}", v, 1);
  }

  clear();
  _createGroundPatch();

  // Retrieve the patch group names
  UniqueArray<String> patch_group_names;
  m_properties->get("PatchGroupNames", patch_group_names);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    info(4) << "Found n=" << patch_group_names.size() << " patchs";

    IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
    for (const String& x : patch_group_names) {
      CellGroup group = cell_family->findGroup(x);
      if (group.null())
        ARCANE_FATAL("Can not find cell group '{0}'", x);
      _addPatchAfterRestore(group);
    }
  }
  else {
    UniqueArray<Int32> level;
    UniqueArray<Int32> overlap;
    UniqueArray<Int32> index;
    UniqueArray<CartCoord> min_point;
    UniqueArray<CartCoord> max_point;

    m_properties->get("LevelPatches", level);
    m_properties->get("OverlapSizePatches", overlap);
    m_properties->get("IndexPatches", index);
    m_properties->get("MinPointPatches", min_point);
    m_properties->get("MaxPointPatches", max_point);

    IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();

    // Note: the ground patch was excluded from the save.
    for (Integer pos_in_array = 0; pos_in_array < index.size(); ++pos_in_array) {
      ConstArrayView min(min_point.subConstView(pos_in_array * 3, 3));
      ConstArrayView max(max_point.subConstView(pos_in_array * 3, 3));

      AMRPatchPosition position(
      level[pos_in_array],
      { min[MD_DirX], min[MD_DirY], min[MD_DirZ] },
      { max[MD_DirX], max[MD_DirY], max[MD_DirZ] },
      overlap[pos_in_array]);

      const String& x = patch_group_names[pos_in_array];
      CellGroup cell_group = cell_family->findGroup(x);
      if (cell_group.null())
        ARCANE_FATAL("Can not find cell group '{0}'", x);

      auto* cdi = new CartesianMeshPatch(m_cmesh, index[pos_in_array], position);
      _addPatchInstance(makeRef(cdi));
      _addCellGroup(cell_group, cdi, true);
    }

    UniqueArray<Int32> available_index;
    m_properties->get("PatchGroupNamesAvailable", available_index);
    rebuildAvailableGroupIndex(available_index);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<CartesianMeshPatch> CartesianPatchGroup::
groundPatch()
{
  _createGroundPatch();
  return patch(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
addPatch(ConstArrayView<Int32> cells_local_id)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Do not use this method with AMR type 3");
  }

  Integer index = _nextIndexForNewPatch();
  String children_group_name = String("CartesianMeshPatchCells") + index;
  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  CellGroup children_cells = cell_family->createGroup(children_group_name, cells_local_id, true);
  _addPatch(children_cells, index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// rebuildAvailableIndex() must be called after calls to this method.
Integer CartesianPatchGroup::
_addPatchAfterRestore(CellGroup cell_group)
{
  const String& name = cell_group.name();
  Integer group_index = -1;
  if (name.startsWith("CartesianMeshPatchCells")) {
    String index_str = name.substring(23);
    group_index = std::stoi(index_str.localstr());
  }
  else {
    ARCANE_FATAL("Invalid group");
  }

  _addPatch(cell_group, group_index);
  return group_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addPatch(CellGroup cell_group, Integer group_index)
{
  _createGroundPatch();
  if (group_index == -1) {
    return;
  }
  if (cell_group.null())
    ARCANE_FATAL("Null cell group");

  AMRPatchPosition position;

  auto* cdi = new CartesianMeshPatch(m_cmesh, group_index, position);
  _addPatchInstance(makeRef(cdi));
  _addCellGroup(cell_group, cdi, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
addPatch(const AMRZonePosition& zone_position)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }

  Trace::Setter mci(traceMng(), "CartesianPatchGroup");

  info() << "addPatch() with zone"
         << " -- Position : " << zone_position.position()
         << " -- Length : " << zone_position.length();

  clearRefineRelatedFlags();

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // The conversion gives us the intermediate patch (which surrounds the cells
  // to be refined, not the refined cells). A call to the AMRPatchPosition::patchUp() method
  // allows the patch to be promoted one level so that it groups the refined cells, thus becoming a "classic" patch.
  AMRPatchPosition position = zone_position.toAMRPatchPosition(m_cmesh);

  Int32 level = position.level();
  Int32 level_up = level + 1;
  Int32 nb_overlap_cells = 0;

  Int32 higher_level = m_higher_level;

  // The patchUp() method will need the future higher_level to correctly calculate
  // the number of overlap layers.
  // If we have a patch that will be higher than all others.
  if (level_up >= higher_level) {
    higher_level = level_up;
    // The number of layers for the highest level must be
    // m_size_of_overlap_layer_top_level. We are on an intermediate patch,
    // so we divide this number by the number of child cells that will be created.
    nb_overlap_cells = m_size_of_overlap_layer_top_level / numbering->pattern();
    debug() << "Higher level -- Old : " << m_higher_level << " -- New : " << higher_level;
  }

  else {
    // The number of overlap layers.
    // +1 because the created patch will be at level level + 1.
    // /pattern because the cells to be refined are at level level.
    //
    // Explanation:
    //  level=0,
    //  the future patch will be at level 1, so the number of overlap layers must be that corresponding to level 1
    //  (i.e., level+1),
    //  but the cells to be refined are at level 0, so we must divide the number of layers by the number of child cells that will be created
    //  (for one dimension) (i.e., numbering->pattern()).
    nb_overlap_cells = overlapLayerSize(level + 1) / numbering->pattern();
  }
  position.setOverlapLayerSize(nb_overlap_cells);

  debug() << "Zone to intermediary patch"
          << " -- minPoint : " << position.minPoint()
          << " -- maxPoint : " << position.maxPoint()
          << " -- overlapLayerSize : " << position.overlapLayerSize()
          << " -- level : " << level;

  // Reminder: only cells with the "II_InPatch" flag can be refined.
  //          This condition is checked in the method
  //          AMRZonePosition::toAMRPatchPosition() called above.
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
    if (!icell->hasHChildren()) {
      const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
      if (position.isInWithOverlap(pos)) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
    }
  }

  amr->refine();

  // We transition from an intermediate patch to a classic patch.
  AMRPatchPosition position_up = position.patchUp(m_cmesh->mesh()->dimension(), higher_level, m_size_of_overlap_layer_top_level);

  info() << "Zone to Patch"
         << " -- minPoint : " << position_up.minPoint()
         << " -- maxPoint : " << position_up.maxPoint()
         << " -- overlapLayerSize : " << position_up.overlapLayerSize()
         << " -- level : " << position_up.level();

  // We add this patch to our object.
  _addPatch(position_up);

  // If the created patch is higher than the others, a new refinement level has
  // been created and we must update the number oF overlap layers for lower levels.
  _updateHigherLevel();

#ifdef ARCANE_CHECK
  _checkPatchesAndMesh();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
nbPatch() const
{
  return m_amr_patches.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<CartesianMeshPatch> CartesianPatchGroup::
patch(const Integer index) const
{
  return m_amr_patches[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshPatchListView CartesianPatchGroup::
patchListView() const
{
  return CartesianMeshPatchListView{ m_amr_patches_pointer };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CartesianPatchGroup::
allCells(const Integer index)
{
  return m_amr_patch_cell_groups_all[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CartesianPatchGroup::
inPatchCells(Integer index)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  return m_amr_patch_cell_groups_inpatch[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CartesianPatchGroup::
overlapCells(Integer index)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  return m_amr_patch_cell_groups_overlap[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Warning: also clears the ground patch. It is necessary to retrieve it afterwards.
// (the m_all_items_direction_info of CartesianMeshImpl)
void CartesianPatchGroup::
clear()
{
  _removeAllPatches();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
removePatch(const Integer index)
{
  if (m_patches_to_delete.contains(index)) {
    return;
  }
  if (index == 0) {
    ARCANE_FATAL("You cannot remove ground patch");
  }
  if (index < 1 || index >= m_amr_patches.size()) {
    ARCANE_FATAL("Invalid index");
  }

  m_patches_to_delete.add(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR Cell");
  }

  // In AMR type 1, there are only "all" groups.
  for (Integer i = 1; i < m_amr_patch_cell_groups_all.size(); ++i) {
    allCells(i).removeItems(cells_local_id);
  }
  applyPatchEdit(true, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removeCellsInAllPatches(const AMRPatchPosition& zone_to_delete)
{
  // Caution if removing the two-step removal: _removePartOfPatch() also deletes patches.
  // i = 1 because the ground patch cannot be defined/refined.
  const Integer nb_patchs = m_amr_patches_pointer.size();
  for (Integer i = 1; i < nb_patchs; ++i) {
    ICartesianMeshPatch* patch = m_amr_patches_pointer[i];
    // info() << "I : " << i
    //                                     << " -- Compare Patch (min : " << patch->position().minPoint()
    //                                     << ", max : " << patch->position().maxPoint()
    //                                     << ", level : " << patch->position().level()
    //                                     << ") and Zone (min : " << zone_to_delete.minPoint()
    //                                     << ", max : " << zone_to_delete.maxPoint()
    //                                     << ", level : " << zone_to_delete.level() << ")";

    if (zone_to_delete.haveIntersection(patch->position())) {
      _removePartOfPatch(i, zone_to_delete);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
removeCellsInZone(const AMRZonePosition& zone_to_delete)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  clearRefineRelatedFlags();

  UniqueArray<Int32> cells_local_id;

  AMRPatchPosition patch_position;
  zone_to_delete.cellsInPatch(m_cmesh, cells_local_id, patch_position);

  _removeCellsInAllPatches(patch_position);
  applyPatchEdit(false, false);
  _updatePatchFlagsOfItemsLevel(patch_position.level(), true);
  _updateHigherLevel();
  _coarsenUselessCells(true);

#ifdef ARCANE_CHECK
  _checkPatchesAndMesh();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
applyPatchEdit(bool remove_empty_patches, bool update_higher_level)
{
  // m_cmesh->mesh()->traceMng()->info() << "applyPatchEdit() -- Remove nb patch : " << m_patches_to_delete.size();

  std::stable_sort(m_patches_to_delete.begin(), m_patches_to_delete.end(),
                   [](const Integer a, const Integer b) {
                     return a < b;
                   });

  _removeMultiplePatches(m_patches_to_delete);
  m_patches_to_delete.clear();

  // In AMR type 3, there cannot be an empty patch.
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly && remove_empty_patches) {
    UniqueArray<Integer> size_of_patches(m_amr_patch_cell_groups_all.size());
    for (Integer i = 0; i < m_amr_patch_cell_groups_all.size(); ++i) {
      size_of_patches[i] = m_amr_patch_cell_groups_all[i].size();
    }
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, size_of_patches);
    for (Integer i = 0; i < size_of_patches.size(); ++i) {
      if (size_of_patches[i] == 0) {
        m_patches_to_delete.add(i);
      }
    }
    _removeMultiplePatches(m_patches_to_delete);
    m_patches_to_delete.clear();
  }

  // In AMR type 1, there is no concept of patch level.
  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly && update_higher_level) {
    _updateHigherLevel();
    _coarsenUselessCells(true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
updateLevelsAndAddGroundPatch()
{
  // TODO: This method should call createSubLevel(), not the other way around.
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    return;
  }
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();

  amr->createSubLevel();

  // Warning: we assume that numbering->updateFirstLevel(); has already been called!

  for (ICartesianMeshPatch* patch : m_amr_patches_pointer) {
    const Int32 level = patch->position().level();
    // If the level is 0, it is the special patch 0, so we only modify the max, the level remains 0.
    if (level == 0) {
      const CartCoord3 max_point = patch->position().maxPoint();
      if (m_cmesh->mesh()->dimension() == 2) {
        patch->_internalApi()->positionRef().setMaxPoint({
        numbering->offsetLevelToLevel(max_point.x, level, level - 1),
        numbering->offsetLevelToLevel(max_point.y, level, level - 1),
        1,
        });
      }
      else {
        patch->_internalApi()->positionRef().setMaxPoint({
        numbering->offsetLevelToLevel(max_point.x, level, level - 1),
        numbering->offsetLevelToLevel(max_point.y, level, level - 1),
        numbering->offsetLevelToLevel(max_point.z, level, level - 1),
        });
      }
    }
    // Otherwise, we "elevate" the level of the patches since there will be the "-1" patch
    else {
      patch->_internalApi()->positionRef().setLevel(level + 1);
    }
  }

  {
    AMRPatchPosition old_ground;
    old_ground.setLevel(1);
    old_ground.setMinPoint({ 0, 0, 0 });
    old_ground.setMaxPoint({ numbering->globalNbCellsX(1), numbering->globalNbCellsY(1), numbering->globalNbCellsZ(1) });
    old_ground.computeOverlapLayerSize(m_higher_level + 1, m_size_of_overlap_layer_top_level);
    _addPatch(old_ground);
  }
  // Calculate the directions of the new ground patch.
  m_cmesh->computeDirectionsPatchV2(0);

  _updatePatchFlagsOfItemsGroundLevel();

  // The methods _increaseOverlapSizeLevel() and _reduceOverlapSizeLevel()
  // will handle recalculating the directions of the elevated patches.
  // Note: Recalculation is necessary because we add/delete overlap cells.
  _updateHigherLevel();

#ifdef ARCANE_CHECK
  _checkPatchesAndMesh();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
_nextIndexForNewPatch()
{
  if (!m_available_group_index.empty()) {
    const Integer elem = m_available_group_index.back();
    m_available_group_index.popBack();
    return elem;
  }
  return m_index_new_patches++;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
mergePatches()
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    return;
  }
  UniqueArray<Int32> new_merged_patches;

  // info() << "Global fusion";
  UniqueArray<std::pair<Integer, Int64>> index_n_nb_cells;
  {
    Integer index = 0;
    for (auto patch : m_amr_patches_pointer) {
      index_n_nb_cells.add({ index++, patch->position().nbCells() });
    }
  }

  // Merge algorithm.
  // First, we sort the patches from the smallest number of cells to the largest number of cells (optional).
  // Then, for each patch, we check if it can be merged with another.
  // If a merge is made, we restart the algorithm until no more merges can be made.
  bool fusion = true;
  while (fusion) {
    fusion = false;

    std::stable_sort(index_n_nb_cells.begin(), index_n_nb_cells.end(),
                     [](const std::pair<Integer, Int64>& a, const std::pair<Integer, Int64>& b) {
                       return a.second < b.second;
                     });

    for (Integer p0 = 0; p0 < index_n_nb_cells.size(); ++p0) {
      auto [index_p0, nb_cells_p0] = index_n_nb_cells[p0];

      AMRPatchPosition& patch_fusion_0 = m_amr_patches_pointer[index_p0]->_internalApi()->positionRef();
      if (patch_fusion_0.isNull())
        continue;

      // If a merge has already occurred, we must then look at the patches before "p0"
      // (since at least one has been modified).
      // (an "optimization" could be to retrieve the position of the first
      // merged patch but well, less readable + not many patches).
      for (Integer p1 = p0 + 1; p1 < m_amr_patches_pointer.size(); ++p1) {
        auto [index_p1, nb_cells_p1] = index_n_nb_cells[p1];

        AMRPatchPosition& patch_fusion_1 = m_amr_patches_pointer[index_p1]->_internalApi()->positionRef();
        if (patch_fusion_1.isNull())
          continue;

        // info() << "\tCheck fusion"
        //                                     << " -- 0 Min point : " << patch_fusion_0.minPoint()
        //                                     << " -- 0 Max point : " << patch_fusion_0.maxPoint()
        //                                     << " -- 0 Level : " << patch_fusion_0.level()
        //                                     << " -- 1 Min point : " << patch_fusion_1.minPoint()
        //                                     << " -- 1 Max point : " << patch_fusion_1.maxPoint()
        //                                     << " -- 1 Level : " << patch_fusion_1.level();

        if (patch_fusion_0.fusion(patch_fusion_1)) {
          // info() << "Fusion OK";
          index_n_nb_cells[p0].second = patch_fusion_0.nbCells();

          // info() << "Remove patch : " << index_p1;
          removePatch(index_p1);

          if (!new_merged_patches.contains(index_p0)) {
            new_merged_patches.add(index_p0);
          }

          auto find_p1 = new_merged_patches.span().findFirst(index_p1);
          if (find_p1.has_value()) {
            new_merged_patches.remove(find_p1.value());
          }

          fusion = true;
          break;
        }
      }
      if (fusion) {
        break;
      }
    }
  }

  UniqueArray<Int32> levels_edited;
  for (Int32 patch_index : new_merged_patches) {
    _updateCellGroups(patch_index, false);

    Int32 level = patch(patch_index)->_internalApi()->positionRef().level();
    if (!levels_edited.contains(level)) {
      levels_edited.add(level);
    }
  }
  applyPatchEdit(false, false);

  for (Int32 level : levels_edited) {
    _updatePatchFlagsOfItemsLevel(level, true);
  }

#ifdef ARCANE_CHECK
  _checkPatchesAndMesh();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
beginAdaptMesh(Int32 nb_levels, Int32 level_to_refine_first)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (m_latest_call_level != -2) {
    ARCANE_FATAL("Call endAdaptMesh() before restart mesh adaptation");
  }
  if (level_to_refine_first > m_higher_level) {
    ARCANE_FATAL("Cannot begin to refine level higher than the actual higher level -- Level to refine first : {0} -- Higher level : {1}", level_to_refine_first, m_higher_level);
  }

  Trace::Setter mci(traceMng(), "CartesianPatchGroup");
  info() << "Begin adapting mesh with higher level = " << (nb_levels - 1);

  // We delete all patches above the first level to refine.
  Int32 max_level = 0;
  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level > level_to_refine_first) {
      removePatch(p);
      max_level = level;
    }
  }
  applyPatchEdit(false, false);

  // We also remove the II_InPatch and II_Overlap flags from the cells so that
  // those that are no longer used in any of the new patches are deleted in the finalizeAdaptMesh() method.
  for (Integer l = level_to_refine_first + 1; l <= max_level; ++l) {
    _removePatchFlagsOfItemsLevel(l);
  }
  // We must adapt all levels below the level to adapt.
  // Patches of level "level_to_refine_first" (exclusive) and higher will be deleted.
  if (nb_levels - 1 != m_higher_level) {
    debug() << "beginAdaptMesh() -- Change overlap layer size -- Old higher level : " << m_higher_level
            << " -- Asked higher level : " << (nb_levels - 1)
            << " -- Adapt level lower than : " << level_to_refine_first;

    for (Int32 level = 1; level <= level_to_refine_first; ++level) {
      _changeOverlapSizeLevel(level, m_higher_level, nb_levels - 1);
    }
  }

  m_target_nb_levels = nb_levels;
  m_latest_call_level = level_to_refine_first - 1;
  // m_higher_level keeps its old value.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
endAdaptMesh()
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (m_latest_call_level == -2) {
    ARCANE_FATAL("Call beginAdaptMesh() before");
  }
  Trace::Setter mci(traceMng(), "CartesianPatchGroup");
  info() << "Finalizing adapting mesh with higher level = " << (m_target_nb_levels - 1);

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();

  // The highest level becomes the last adapted level (+1 to have the refined level).
  // We are sure this is the highest level given that we systematically delete
  // patches above it in adaptLevel().
  // beginAdaptMesh() sets the first level to refine given by the user
  // in the m_latest_call_level attribute and checks if the level exists. The
  // first m_latest_call_level is therefore valid.
  m_higher_level = m_latest_call_level + 1;

  // If m_latest_call_level == 0, then adaptLevel() created level 1, so
  // there are 2 levels.
  // If the highest created level is lower than the highest level given
  // by the user in the beginAdaptMesh() method, we are forced to
  // re-adapt the number of overlap cell layers for each patch.
  if (m_higher_level + 1 < m_target_nb_levels) {
    info() << "Reduce higher level from " << (m_target_nb_levels - 1) << " to " << m_higher_level;

    for (Int32 level = 1; level <= m_higher_level; ++level) {
      _changeOverlapSizeLevel(level, m_target_nb_levels - 1, m_higher_level);
    }
  }

  _coarsenUselessCells(true);

  m_target_nb_levels = 0;
  m_latest_call_level = -2;
  clearRefineRelatedFlags();

  info() << "Patch list:";

  for (Integer i = 0; i <= m_higher_level; ++i) {
    for (auto p : m_amr_patches_pointer) {
      auto& position = p->_internalApi()->positionRef();
      if (position.level() == i) {
        info() << "\tPatch #" << p->index()
               << " -- Level : " << position.level()
               << " -- Min point : " << position.minPoint()
               << " -- Max point : " << position.maxPoint()
               << " -- Overlap layer size : " << position.overlapLayerSize();
      }
    }
  }
#ifdef ARCANE_CHECK
  _checkPatchesAndMesh();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
adaptLevel(Int32 level_to_adapt, bool do_fatal_if_useless)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (m_latest_call_level == -2) {
    ARCANE_FATAL("Call beginAdaptMesh() before to begin a mesh adaptation");
  }
  if (level_to_adapt + 1 >= m_target_nb_levels || level_to_adapt < 0) {
    ARCANE_FATAL("Bad level to adapt -- Level to adapt : {0} (creating level {1}) -- Max nb levels : {2}", level_to_adapt, level_to_adapt + 1, m_target_nb_levels);
  }

  Trace::Setter mci(traceMng(), "CartesianPatchGroup");

  if (level_to_adapt > m_latest_call_level + 1) {
    if (do_fatal_if_useless) {
      ARCANE_FATAL("You must refine level {0} before.", (m_latest_call_level + 1));
    }
    warning() << String::format("Useless call -- You must refine level {0} before.", (m_latest_call_level + 1));
    return;
  }

  // We delete all patches above the level we want to adapt.
  // We also do this here in the case where the user calls this method
  // with a level lower than their previous call (which is not
  // necessarily optimal since we delete what was calculated
  // previously...).
  if (level_to_adapt < m_latest_call_level) {
    Int32 max_level = 0;
    for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
      Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
      if (level > level_to_adapt) {
        removePatch(p);
        max_level = level;
      }
    }
    applyPatchEdit(false, false);

    for (Integer l = level_to_adapt + 1; l <= max_level; ++l) {
      _removePatchFlagsOfItemsLevel(l);
    }
  }

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // The number of overlap cell layers.
  // +1 because the created patches will be at level level_to_adapt + 1.
  // /pattern because the cells to refine are at level level_to_adapt.
  //
  // Explanation:
  //  level_to_adapt=0,
  //  future patches will be at level 1, so the number of overlap layers
  //  must be that corresponding to level 1
  //  (thus level_to_adapt+1),
  //  or, the cells to refine are at level 0, so we must divide the
  //  number of layers by the number of child cells that will be created
  //  (for one dimension) (thus numbering->pattern()).
  Int32 nb_overlap_cells = overlapLayerSize(level_to_adapt + 1) / numbering->pattern();

  info() << "adaptLevel()"
         << " -- Level to adapt : " << level_to_adapt
         << " -- Nb of overlap cells (intermediary patch) : " << nb_overlap_cells;

  // Two checks:
  // - we cannot refine multiple levels at once,
  // - we cannot refine cells that are not in a patch (overlap cells are not necessarily in a patch).
  // In addition, we must know if there is at least one cell with the
  // II_Refine flag to know if it is useful to continue the method or not.
  bool has_cell_to_refine = false;
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
    if (icell->hasFlags(ItemFlags::II_Refine)) {
      if (icell->level() != level_to_adapt) {
        ARCANE_FATAL("Flag II_Refine found on Cell (UID={0} - Level={1}) not in level to refine (={2})", icell->uniqueId(), icell->level(), level_to_adapt);
      }
      if (level_to_adapt != 0 && !icell->hasFlags(ItemFlags::II_InPatch)) {
        const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
        ARCANE_FATAL("Cannot refine cell not in patch -- Pos : {0} -- CellUID : {1} -- CellLevel : {2}", pos, icell->uniqueId(), icell->level());
      }
      has_cell_to_refine = true;
    }
  }
  has_cell_to_refine = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, has_cell_to_refine);

  if (!has_cell_to_refine) {
    if (do_fatal_if_useless) {
      ARCANE_FATAL("There are no cells to refine.");
    }
    // We recall that, in endAdaptMesh(), m_higher_level will take the value
    // of m_latest_call_level + 1.
    //
    // It is important to set -1 here in the case where (for example):
    // - Initially, there are no patches (m_latest_call_level == -1), adaptLevel(0):
    //  -> We refine level 0 cells, so m_latest_call_level = 0 and
    //  thus, m_higher_level will be equal to 1
    // - We call adaptLevel(0) a second time:
    //  -> The user has not marked any level 0 cells, so
    //  m_latest_call_level = -1 and thus m_higher_level = 0.
    // Above, we delete levels higher than level_to_adapt even if
    // the call to adaptLevel() does not refine any cells, so we must update
    // m_latest_call_level.
    m_latest_call_level = level_to_adapt - 1;
    debug() << "adaptLevel() -- End call -- No refine -- Actual patch list:";

    for (Integer i = 0; i <= m_target_nb_levels; ++i) {
      for (auto p : m_amr_patches_pointer) {
        auto& position = p->_internalApi()->positionRef();
        if (position.level() == i) {
          debug() << "\tPatch #" << p->index()
                  << " -- Level : " << position.level()
                  << " -- Min point : " << position.minPoint()
                  << " -- Max point : " << position.maxPoint()
                  << " -- Overlap layer size : " << position.overlapLayerSize();
        }
      }
    }
    return;
  }

  m_latest_call_level = level_to_adapt;

  UniqueArray<AMRPatchPositionSignature> sig_array;

  // We must provide one or more initial patches to be reduced and
  // cut.
  // If the level to adapt is level 0, we can create an initial patch
  // that is the size of the ground patch.
  // We don't need to reduce it; AMRPatchPositionSignature::fillSig()
  // will take care of it.
  if (level_to_adapt == 0) {
    AMRPatchPosition all_level;
    all_level.setLevel(level_to_adapt);
    all_level.setMinPoint({ 0, 0, 0 });
    all_level.setMaxPoint({ numbering->globalNbCellsX(level_to_adapt), numbering->globalNbCellsY(level_to_adapt), numbering->globalNbCellsZ(level_to_adapt) });
    // For this setOverlapLayerSize(), see the explanation above.
    all_level.setOverlapLayerSize(nb_overlap_cells);
    AMRPatchPositionSignature sig(all_level, m_cmesh);
    sig_array.add(sig);
  }

  // For other levels, we create the initial patches by copying the
  // patches from level_to_adapt.
  // We cannot create a patch that is the size of level_to_adapt
  // because it is imperative that the patch(es) generated by
  // AMRPatchPositionSignatureCut (future patches of level_to_adapt+1)
  // be included in the patch(es) of level_to_adapt! (otherwise we would have orphaned cells).
  // We can take these patches as initial patches because we know that the
  // only cells we will refine are in the patch(es) of level_to_adapt (cells having the II_InPatch flag).
  // We also know that the methods of AMRPatchPositionSignatureCut cannot enlarge the initial patches (only reduce or cut).
  // (and yes, the if(level_to_adapt == 0) is not essential, but since we know there is only one level 0 patch, it is faster).
  else {
    for (auto patch : m_amr_patches_pointer) {
      Integer level = patch->_internalApi()->positionRef().level();
      if (level == level_to_adapt) {
        auto position = patch->position();
        position.setOverlapLayerSize(nb_overlap_cells);
        AMRPatchPositionSignature sig(position, m_cmesh);
        sig_array.add(sig);
      }
    }
  }

  AMRPatchPositionSignatureCut::cut(sig_array);

  // Once the patches are cut, we add the II_Refine flag to the cells of
  // these patches.
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt)) {
    if (!icell->hasHChildren()) {
      const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
      for (const AMRPatchPositionSignature& patch_signature : sig_array) {
        if (patch_signature.patch().isInWithOverlap(pos)) {
          if (!icell->hasFlags(ItemFlags::II_InPatch) && !icell->hasFlags(ItemFlags::II_Overlap)) {
            ARCANE_FATAL("Internal error -- Refine algo error -- Pos : {0}", pos);
          }
          icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
        }
      }
    }
  }

  {
    //   UniqueArray<CartCoord> out(numbering->globalNbCellsY(level_to_adapt) * numbering->globalNbCellsX(level_to_adapt), -1);
    //   Array2View av_out(out.data(), numbering->globalNbCellsY(level_to_adapt), numbering->globalNbCellsX(level_to_adapt));
    //   ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt)) {
    //     CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    //     if (icell->hasHChildren()) {
    //       av_out(pos.y, pos.x) = 0;
    //     }
    //     if (icell->hasFlags(ItemFlags::II_Refine)) {
    //       av_out(pos.y, pos.x) = 1;
    //     }
    //   }
    //
    //   StringBuilder str = "";
    //   for (CartCoord i = 0; i < numbering->globalNbCellsX(level_to_adapt); ++i) {
    //     str += "\n";
    //     for (CartCoord j = 0; j < numbering->globalNbCellsY(level_to_adapt); ++j) {
    //       CartCoord c = av_out(i, j);
    //       if (c == 1)
    //         str += "[++]";
    //       else if (c == 0)
    //         str += "[XX]";
    //       else
    //         str += "[  ]";
    //     }
    //   }
    //   info() << str;
  }

  // We refine.
  amr->refine();

  // TODO: Normally, this is not necessary, should be corrected in amr->refine().
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_Refine);
  }

  // The patches in sig_array are "intermediate" patches. They are patches of level_to_adapt representing patches of level_to_adapt+1.
  // It is now necessary to convert them into level_to_adapt+1 patches.
  UniqueArray<AMRPatchPosition> all_patches;
  for (const auto& elem : sig_array) {
    all_patches.add(elem.patch().patchUp(m_cmesh->mesh()->dimension(), m_target_nb_levels - 1, m_size_of_overlap_layer_top_level));
  }

  // We merge the patches that can be merged before "adding" them to the mesh.
  AMRPatchPositionLevelGroup::fusionPatches(all_patches, true);

  // for (const AMRPatchPosition& patch : all_patches) {
  //   debug() << "\tPatch AAA"
  //           << " -- Level : " << patch.level()
  //           << " -- Min point : " << patch.minPoint()
  //           << " -- Max point : " << patch.maxPoint()
  //           << " -- overlapLayerSize : " << patch.overlapLayerSize();
  // }

  {
    //   UniqueArray<CartCoord> out(numbering->globalNbCellsY(level_to_adapt + 1) * numbering->globalNbCellsX(level_to_adapt + 1), -1);
    //   Array2View av_out(out.data(), numbering->globalNbCellsY(level_to_adapt + 1), numbering->globalNbCellsX(level_to_adapt + 1));
    //   ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt + 1)) {
    //
    //     CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    //     Integer patch = -1;
    //
    //     if (icell->hasFlags(ItemFlags::II_Overlap) && icell->hasFlags(ItemFlags::II_InPatch)) {
    //       patch = -2;
    //     }
    //     else if (icell->hasFlags(ItemFlags::II_Overlap)) {
    //       patch = -3;
    //     }
    //     else if (icell->hasFlags(ItemFlags::II_InPatch)) {
    //       patch = -4;
    //     }
    //     else {
    //       patch = -5;
    //     }
    //     av_out(pos.y, pos.x) = patch;
    //     if (icell->uniqueId() == 3310) {
    //       info() << "Maille présente ! -- Coord : " << pos << " -- Flags : " << patch;
    //     }
    //   }
    //
    //   StringBuilder str = "";
    //   for (CartCoord i = 0; i < numbering->globalNbCellsX(level_to_adapt + 1); ++i) {
    //     str += "\n";
    //     for (CartCoord j = 0; j < numbering->globalNbCellsY(level_to_adapt + 1); ++j) {
    //       CartCoord c = av_out(i, j);
    //       if (c >= 0) {
    //         str += "[";
    //         if (c < 10)
    //           str += " ";
    //         str += c;
    //         str += "]";
    //       }
    //       else if (c == -2) {
    //         str += "[OI]";
    //       }
    //       else if (c == -3) {
    //         str += "[OO]";
    //       }
    //       else if (c == -4) {
    //         str += "[II]";
    //       }
    //       else if (c == -5) {
    //         str += "[XX]";
    //       }
    //       else
    //         str += "[  ]";
    //     }
    //   }
    //   info() << str;
  }

  // We add the patches to the mesh (we create the groups) and calculate the
  // directions for each of them (so that the user can use them directly).
  for (const AMRPatchPosition& patch : all_patches) {
    Integer index = _addPatch(patch);
    // TODO: But that is not a good idea at all!
    m_cmesh->computeDirectionsPatchV2(index);
  }

  debug() << "adaptLevel() -- End call -- Actual patch list:";

  for (Integer i = 0; i <= m_target_nb_levels; ++i) {
    for (auto p : m_amr_patches_pointer) {
      auto& position = p->_internalApi()->positionRef();
      if (position.level() == i) {
        debug() << "\tPatch #" << p->index()
                << " -- Level : " << position.level()
                << " -- Min point : " << position.minPoint()
                << " -- Max point : " << position.maxPoint()
                << " -- Overlap layer size : " << position.overlapLayerSize();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_increaseOverlapSizeLevel(Int32 level_to_increate, Int32 new_size)
{
  if (level_to_increate == 0) {
    ARCANE_FATAL("Level 0 has not overlap layer");
  }

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  bool has_cell_to_refine = false;

  // Three main steps:
  // - first, we increase the number of layers in the position structures,
  // then we add the II_Refine flag to parent cells that have no children,
  // - we refine the cells,
  // - we add the flags to the new cells and add them to the cell groups
  // of the patches.
  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_increate) {
      AMRPatchPosition& position = m_amr_patches_pointer[p]->_internalApi()->positionRef();

      Int32 size_layer = position.overlapLayerSize();
      if (size_layer > new_size) {
        ARCANE_FATAL("Cannot reduce layer with _increaseOverlapSizeLevel method");
      }

      // We could check that the number of layers of all patches of a level is identical.

      if (size_layer == new_size) {
        continue;
      }

      has_cell_to_refine = true;
      position.setOverlapLayerSize(new_size);

      // The cells to refine are on the lower level.
      // For each cell, to know whether we should refine it or not, we
      // move it up one level and check if it is in the overlap layers.
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate - 1)) {
        const CartCoord3 pos = numbering->offsetLevelToLevel(numbering->cellUniqueIdToCoord(*icell), level_to_increate - 1, level_to_increate);
        if (position.isInWithOverlap(pos) && !icell->hasHChildren()) {
          icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
        }
      }
    }
  }
  has_cell_to_refine = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, has_cell_to_refine);
  if (!has_cell_to_refine) {
    return;
  }

  // We refine the cells.
  amr->refine();

  // TODO: Normally, this is not necessary, should be corrected in amr->refine().
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate - 1)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_Refine);
  }

  UniqueArray<Int32> cell_to_add;

  // Add the flags and update the cell groups of the patches.
  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_increate) {
      AMRPatchPosition& position = m_amr_patches_pointer[p]->_internalApi()->positionRef();

      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate)) {
        if (!icell->hasFlags(ItemFlags::II_JustAdded))
          continue;

        const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);

        if (position.isInWithOverlap(pos)) {
          cell_to_add.add(icell.localId());
          icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
          for (Face face : icell->faces()) {
            face.mutableItemBase().addFlags(ItemFlags::II_Overlap);
          }
          for (Node node : icell->nodes()) {
            node.mutableItemBase().addFlags(ItemFlags::II_Overlap);
          }
        }
        else if (position.isIn(pos)) {
          icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
          for (Face face : icell->faces()) {
            face.mutableItemBase().addFlags(ItemFlags::II_InPatch);
          }
          for (Node node : icell->nodes()) {
            node.mutableItemBase().addFlags(ItemFlags::II_InPatch);
          }
        }
      }

      allCells(p).addItems(cell_to_add, true); //TODO Normally, set check = false
      overlapCells(p).addItems(cell_to_add, true);
      cell_to_add.clear();

      // We calculate the directions so that the patch can be used.
      m_cmesh->computeDirectionsPatchV2(p);
    }
  }

  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_JustAdded);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_reduceOverlapSizeLevel(Int32 level_to_reduce, Int32 new_size)
{
  // Warning: The reduction is possible because we are not deleting cells, we are removing their InPatch/Overlap flags so that they can be deleted later.
  // We only remove them from the cell groups of the patches.
  // It is therefore necessary to have another method after this one to delete cells without flags.

  if (level_to_reduce == 0) {
    ARCANE_FATAL("Level 0 has not overlap layer");
  }

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // Two steps:
  // - first, we update the position structures of the patches, then we
  // remove from the cell groups of the patches the cells that are no longer
  // in the overlap layers,
  // - finally, we recalculate the flags for the entire level.
  bool has_cell_to_mark = false;
  UniqueArray<Int32> cell_to_remove;

  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_reduce) {
      AMRPatchPosition& position = m_amr_patches_pointer[p]->_internalApi()->positionRef();

      Int32 size_layer = position.overlapLayerSize();
      if (size_layer < new_size) {
        ARCANE_FATAL("Cannot add layer with _reduceOverlapSizeLevel method");
      }
      if (size_layer == new_size) {
        continue;
      }

      has_cell_to_mark = true;
      position.setOverlapLayerSize(new_size);

      ENUMERATE_ (Cell, icell, overlapCells(p)) {
        const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
        if (!position.isInWithOverlap(pos)) {
          cell_to_remove.add(icell.localId());
        }
      }

      allCells(p).removeItems(cell_to_remove, true); //TODO Normally, set check = false
      overlapCells(p).removeItems(cell_to_remove, true);
      cell_to_remove.clear();
    }
  }
  has_cell_to_mark = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, has_cell_to_mark);
  if (!has_cell_to_mark) {
    return;
  }

  // Because of the mixing of the two flags, we must recalculate the flags.
  _updatePatchFlagsOfItemsLevel(level_to_reduce, true);

  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_reduce) {
      m_cmesh->computeDirectionsPatchV2(p);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_updateHigherLevel()
{
  // We check which is the highest patch.
  Int32 higher_level_patch = 0;

  for (const auto patch : m_amr_patches_pointer) {
    const Int32 level = patch->_internalApi()->positionRef().level();
    if (level > higher_level_patch) {
      higher_level_patch = level;
    }
  }

  if (higher_level_patch != m_higher_level) {
    for (Int32 level = 1; level <= higher_level_patch; ++level) {
      _changeOverlapSizeLevel(level, m_higher_level, higher_level_patch);
    }

    m_higher_level = higher_level_patch;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_changeOverlapSizeLevel(Int32 level, Int32 previous_higher_level, Int32 new_higher_level)
{
  if (previous_higher_level == new_higher_level) {
    return;
  }

  Int32 old_overlap_size = ((level > previous_higher_level) ? 0 : AMRPatchPosition::computeOverlapLayerSize(level, previous_higher_level, m_size_of_overlap_layer_top_level));
  Int32 new_overlap_size = AMRPatchPosition::computeOverlapLayerSize(level, new_higher_level, m_size_of_overlap_layer_top_level);

  if (old_overlap_size == new_overlap_size) {
    return;
  }
  if (old_overlap_size < new_overlap_size) {
    _increaseOverlapSizeLevel(level, new_overlap_size);
  }
  else {
    _reduceOverlapSizeLevel(level, new_overlap_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_coarsenUselessCells(bool use_cells_level)
{
  Int32 higher_level_patch = m_higher_level;
  if (use_cells_level) {
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
      if (icell->level() > higher_level_patch) {
        higher_level_patch = icell->level();
      }
    }
    higher_level_patch = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, higher_level_patch);
  }

  // We delete the cells that are not/no longer in a patch.
  for (Integer level = higher_level_patch; level > 0; --level) {
    _coarsenUselessCellsInLevel(level);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_coarsenUselessCellsInLevel(Int32 level)
{
  Integer nb_cells_to_coarse = 0;
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
    if (!icell->hasFlags(ItemFlags::II_InPatch) && !icell->hasFlags(ItemFlags::II_Overlap)) {
      //debug() << "Coarse CellUID : " << icell->uniqueId();
      icell->mutableItemBase().addFlags(ItemFlags::II_Coarsen);
      nb_cells_to_coarse++;
    }
  }
  debug() << "Remove " << nb_cells_to_coarse << " refined cells without flag in level " << level;
  nb_cells_to_coarse = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, nb_cells_to_coarse);

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  if (nb_cells_to_coarse != 0) {
    amr->coarsen(true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_updatePatchFlagsOfItemsLevel(Int32 level, bool use_cell_groups)
{
  if (level == 0) {
    _updatePatchFlagsOfItemsGroundLevel();
    return;
  }

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap);
  }
  ENUMERATE_ (Face, iface, m_cmesh->mesh()->allLevelCells(level).faceGroup()) {
    iface->mutableItemBase().removeFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap);
  }
  ENUMERATE_ (Node, inode, m_cmesh->mesh()->allLevelCells(level).nodeGroup()) {
    inode->mutableItemBase().removeFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap);
  }

  // By using the patch cell_groups, we don't need to search,
  // for each mesh, whether it is in each patch.
  // But this requires that the cell_groups are available.
  if (use_cell_groups) {
    for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
      Int32 level_patch = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
      if (level_patch == level) {
        ENUMERATE_ (Cell, icell, inPatchCells(p)) {
          icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
        }
        ENUMERATE_ (Face, iface, inPatchCells(p).faceGroup()) {
          iface->mutableItemBase().addFlags(ItemFlags::II_InPatch);
        }
        ENUMERATE_ (Node, inode, inPatchCells(p).nodeGroup()) {
          inode->mutableItemBase().addFlags(ItemFlags::II_InPatch);
        }

        ENUMERATE_ (Cell, icell, overlapCells(p)) {
          icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
        }
        ENUMERATE_ (Face, iface, overlapCells(p).faceGroup()) {
          iface->mutableItemBase().addFlags(ItemFlags::II_Overlap);
        }
        ENUMERATE_ (Node, inode, overlapCells(p).nodeGroup()) {
          inode->mutableItemBase().addFlags(ItemFlags::II_Overlap);
        }
      }
    }
  }

  // Otherwise, a brute force method that always works.
  else {
    // We add the flags to the patch cells.
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
      bool in_overlap = false;
      bool in_patch = false;

      // If a mesh is in a patch, it gets the II_InPatch flag.
      // If a mesh is an overlap mesh for a patch, it gets
      // the II_Overlap flag.
      // As its name suggests, an overlap mesh can overlap another
      // patch. So a mesh can be both II_InPatch and
      // II_Overlap.
      const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);

      for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
        auto& patch = m_amr_patches_pointer[p]->_internalApi()->positionRef();
        if (patch.level() != level) {
          continue;
        }

        if (patch.isIn(pos)) {
          in_patch = true;
        }
        else if (patch.isInWithOverlap(pos)) {
          in_overlap = true;
        }
        if (in_patch && in_overlap) {
          break;
        }
      }
      if (in_patch && in_overlap) {
        icell->mutableItemBase().addFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap);
        for (Face face : icell->faces()) {
          face.mutableItemBase().addFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap);
        }
        for (Node node : icell->nodes()) {
          node.mutableItemBase().addFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap);
        }
      }
      else if (in_overlap) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
        icell->mutableItemBase().removeFlags(ItemFlags::II_InPatch); //Just in case.
        for (Face face : icell->faces()) {
          face.mutableItemBase().addFlags(ItemFlags::II_Overlap);
        }
        for (Node node : icell->nodes()) {
          node.mutableItemBase().addFlags(ItemFlags::II_Overlap);
        }
      }
      else if (in_patch) {
        icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
        icell->mutableItemBase().removeFlags(ItemFlags::II_Overlap); //Just in case.
        for (Face face : icell->faces()) {
          face.mutableItemBase().addFlags(ItemFlags::II_InPatch);
        }
        for (Node node : icell->nodes()) {
          node.mutableItemBase().addFlags(ItemFlags::II_InPatch);
        }
      }
      else {
        icell->mutableItemBase().removeFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap); //Just in case.
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_updatePatchFlagsOfItemsGroundLevel()
{
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(0)) {
    icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
  }
  ENUMERATE_ (Face, iface, m_cmesh->mesh()->allLevelCells(0).faceGroup()) {
    iface->mutableItemBase().addFlags(ItemFlags::II_InPatch);
  }
  ENUMERATE_ (Node, inode, m_cmesh->mesh()->allLevelCells(0).nodeGroup()) {
    inode->mutableItemBase().addFlags(ItemFlags::II_InPatch);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removePatchFlagsOfItemsLevel(Int32 level)
{
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_Overlap | ItemFlags::II_InPatch);
  }
  ENUMERATE_ (Face, iface, m_cmesh->mesh()->allLevelCells(level).faceGroup()) {
    iface->mutableItemBase().removeFlags(ItemFlags::II_Overlap | ItemFlags::II_InPatch);
  }
  ENUMERATE_ (Node, inode, m_cmesh->mesh()->allLevelCells(level).nodeGroup()) {
    inode->mutableItemBase().removeFlags(ItemFlags::II_Overlap | ItemFlags::II_InPatch);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_checkPatchesAndMesh()
{
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
  {
    Int32 higher_level = 0;
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
      if (icell->level() > higher_level) {
        higher_level = icell->level();
      }
    }
    higher_level = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, higher_level);
    if (higher_level != m_higher_level) {
      ARCANE_FATAL("_checkPatchesAndMesh -- Bad higher level -- m_higher_level : {0} -- Found : {1}", m_higher_level, higher_level);
    }
  }
  {
    for (Int32 level = 0; level < m_higher_level; ++level) {
      Int32 check_overlap = overlapLayerSize(level);
      for (Integer p = 0; p < m_amr_patches_pointer.size(); ++p) {
        auto& position = m_amr_patches_pointer[p]->_internalApi()->positionRef();
        if (position.level() == level) {
          if (check_overlap == -1) {
            check_overlap = position.overlapLayerSize();
          }
          else if (check_overlap != position.overlapLayerSize()) {
            ARCANE_FATAL("_checkPatchesAndMesh -- Overlap size incoherence -- Patch pos : {0} -- Previous size : {1} -- Found : {2}", p, check_overlap, position.overlapLayerSize());
          }
        }
      }
    }
  }
  {
    // II_UserMark1 = II_Overlap
    // II_UserMark2 = II_InPatch
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
      Integer level = icell->level();

      bool in_overlap = false;
      bool in_patch = false;

      const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);

      for (auto p : m_amr_patches_pointer) {
        auto& patch = p->_internalApi()->positionRef();
        if (patch.level() != level) {
          continue;
        }

        if (patch.isIn(pos)) {
          in_patch = true;
        }
        else if (patch.isInWithOverlap(pos)) {
          in_overlap = true;
        }
        if (in_patch && in_overlap) {
          break;
        }
      }
      if (in_patch && in_overlap) {
        icell->mutableItemBase().addFlags(ItemFlags::II_UserMark1); // II_Overlap
        icell->mutableItemBase().addFlags(ItemFlags::II_UserMark2); // II_InPatch
      }
      else if (in_overlap) {
        icell->mutableItemBase().addFlags(ItemFlags::II_UserMark1); // II_Overlap
        icell->mutableItemBase().removeFlags(ItemFlags::II_UserMark2); // II_InPatch
      }
      else if (in_patch) {
        icell->mutableItemBase().addFlags(ItemFlags::II_UserMark2); // II_InPatch
        icell->mutableItemBase().removeFlags(ItemFlags::II_UserMark1); // II_Overlap
      }
      else {
        icell->mutableItemBase().removeFlags(ItemFlags::II_UserMark2); // II_InPatch
        icell->mutableItemBase().removeFlags(ItemFlags::II_UserMark1); // II_Overlap
      }
    }
    ENUMERATE_ (Face, iface, m_cmesh->mesh()->allFaces()) {
      Int32 max_level = 0;
      for (Cell cell : iface->cells()) {
        if (cell.level() > max_level) {
          max_level = cell.level();
        }
      }
      for (Cell cell : iface->cells()) {
        if (cell.level() != max_level) {
          continue;
        }
        if (cell.hasFlags(ItemFlags::II_UserMark1)) {
          iface->mutableItemBase().addFlags(ItemFlags::II_UserMark1); // II_Overlap
        }
        if (cell.hasFlags(ItemFlags::II_UserMark2)) {
          iface->mutableItemBase().addFlags(ItemFlags::II_UserMark2); // II_InPatch
        }
      }
    }
    ENUMERATE_ (Node, inode, m_cmesh->mesh()->allNodes()) {
      Int32 max_level = 0;
      for (Cell cell : inode->cells()) {
        if (cell.level() > max_level) {
          max_level = cell.level();
        }
      }
      for (Cell cell : inode->cells()) {
        if (cell.level() != max_level) {
          continue;
        }
        if (cell.hasFlags(ItemFlags::II_UserMark1)) {
          inode->mutableItemBase().addFlags(ItemFlags::II_UserMark1); // II_Overlap
        }
        if (cell.hasFlags(ItemFlags::II_UserMark2)) {
          inode->mutableItemBase().addFlags(ItemFlags::II_UserMark2); // II_InPatch
        }
      }
    }

    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
      if (icell->hasFlags(ItemFlags::II_UserMark1)) {
        if (!icell->hasFlags(ItemFlags::II_Overlap)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_UserMark1 but not II_Overlap -- CellUID : {0}", icell->uniqueId());
        }
      }
      if (icell->hasFlags(ItemFlags::II_UserMark2)) {
        if (!icell->hasFlags(ItemFlags::II_InPatch)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_UserMark2 but not II_InPatch -- CellUID : {0}", icell->uniqueId());
        }
      }
      if (icell->hasFlags(ItemFlags::II_Overlap)) {
        if (!icell->hasFlags(ItemFlags::II_UserMark1)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_Overlap but not II_UserMark1 -- CellUID : {0}", icell->uniqueId());
        }
      }
      if (icell->hasFlags(ItemFlags::II_InPatch)) {
        if (!icell->hasFlags(ItemFlags::II_UserMark2)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_InPatch but not II_UserMark2 -- CellUID : {0}", icell->uniqueId());
        }
      }

      // Today, we can have refined cells but in no patch.

      icell->mutableItemBase().removeFlags(ItemFlags::II_UserMark1); // II_Overlap
      icell->mutableItemBase().removeFlags(ItemFlags::II_UserMark2); // II_InPatch
    }
    ENUMERATE_ (Face, iface, m_cmesh->mesh()->allFaces()) {
      if (iface->hasFlags(ItemFlags::II_UserMark1)) {
        if (!iface->hasFlags(ItemFlags::II_Overlap)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_UserMark1 but not II_Overlap -- FaceUID : {0}", iface->uniqueId());
        }
      }
      if (iface->hasFlags(ItemFlags::II_UserMark2)) {
        if (!iface->hasFlags(ItemFlags::II_InPatch)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_UserMark2 but not II_InPatch -- FaceUID : {0}", iface->uniqueId());
        }
      }
      if (iface->hasFlags(ItemFlags::II_Overlap)) {
        if (!iface->hasFlags(ItemFlags::II_UserMark1)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_Overlap but not II_UserMark1 -- FaceUID : {0}", iface->uniqueId());
        }
      }
      if (iface->hasFlags(ItemFlags::II_InPatch)) {
        if (!iface->hasFlags(ItemFlags::II_UserMark2)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_InPatch but not II_UserMark2 -- FaceUID : {0}", iface->uniqueId());
        }
      }

      iface->mutableItemBase().removeFlags(ItemFlags::II_UserMark1); // II_Overlap
      iface->mutableItemBase().removeFlags(ItemFlags::II_UserMark2); // II_InPatch
    }
    ENUMERATE_ (Node, inode, m_cmesh->mesh()->allNodes()) {
      if (inode->hasFlags(ItemFlags::II_UserMark1)) {
        if (!inode->hasFlags(ItemFlags::II_Overlap)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_UserMark1 but not II_Overlap -- NodeUID : {0}", inode->uniqueId());
        }
      }
      if (inode->hasFlags(ItemFlags::II_UserMark2)) {
        if (!inode->hasFlags(ItemFlags::II_InPatch)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_UserMark2 but not II_InPatch -- NodeUID : {0}", inode->uniqueId());
        }
      }
      if (inode->hasFlags(ItemFlags::II_Overlap)) {
        if (!inode->hasFlags(ItemFlags::II_UserMark1)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_Overlap but not II_UserMark1 -- NodeUID : {0}", inode->uniqueId());
        }
      }
      if (inode->hasFlags(ItemFlags::II_InPatch)) {
        if (!inode->hasFlags(ItemFlags::II_UserMark2)) {
          ARCANE_FATAL("_checkPatchesAndMesh -- II_InPatch but not II_UserMark2 -- NodeUID : {0}", inode->uniqueId());
        }
      }

      inode->mutableItemBase().removeFlags(ItemFlags::II_UserMark1); // II_Overlap
      inode->mutableItemBase().removeFlags(ItemFlags::II_UserMark2); // II_InPatch
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
clearRefineRelatedFlags() const
{
  constexpr ItemFlags::FlagType flags_to_remove = (ItemFlags::II_Coarsen | ItemFlags::II_Refine |
                                                   ItemFlags::II_JustCoarsened | ItemFlags::II_JustRefined |
                                                   ItemFlags::II_JustAdded | ItemFlags::II_CoarsenInactive);
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
    icell->mutableItemBase().removeFlags(flags_to_remove);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
rebuildAvailableGroupIndex(ConstArrayView<Integer> available_group_index)
{
  m_available_group_index = available_group_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> CartesianPatchGroup::
availableGroupIndex()
{
  return m_available_group_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
setOverlapLayerSizeTopLevel(Int32 size_of_overlap_layer_top_level)
{
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  Int32 new_size_of_overlap_layer_top_level = 0;
  // The value -1 is a special value that allows disabling overlap cells.
  if (size_of_overlap_layer_top_level == -1)
    new_size_of_overlap_layer_top_level = -1;
  else
    // We ensure that the size provided by the user is a multiple of
    // pattern (2 today).
    new_size_of_overlap_layer_top_level = size_of_overlap_layer_top_level + (size_of_overlap_layer_top_level % numbering->pattern());

  if (new_size_of_overlap_layer_top_level == m_size_of_overlap_layer_top_level) {
    return;
  }

  // If there is a change in the size of the top level layer, there
  // will be a size change on other levels.
  for (Int32 level = 1; level <= m_higher_level; ++level) {
    Int32 old_overlap_size = AMRPatchPosition::computeOverlapLayerSize(level, m_higher_level, m_size_of_overlap_layer_top_level);
    Int32 new_overlap_size = AMRPatchPosition::computeOverlapLayerSize(level, m_higher_level, new_size_of_overlap_layer_top_level);

    if (old_overlap_size == new_overlap_size) {
      continue;
    }
    if (old_overlap_size < new_overlap_size) {
      _increaseOverlapSizeLevel(level, new_overlap_size);
    }
    else {
      _reduceOverlapSizeLevel(level, new_overlap_size);
    }
  }
  m_size_of_overlap_layer_top_level = new_size_of_overlap_layer_top_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianPatchGroup::
overlapLayerSize(Int32 level)
{
  if (level == 0) {
    return 0;
  }
  // Two cases:
  // - we are in a refinement phase (beginAdaptMesh()), so we must
  //   consider that the highest level is m_target_nb_levels-1,
  // - otherwise, we take the current highest level.
  Int32 higher_level = m_higher_level;
  if (m_target_nb_levels != 0) {
    higher_level = m_target_nb_levels - 1;
  }
  return AMRPatchPosition::computeOverlapLayerSize(level, higher_level, m_size_of_overlap_layer_top_level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addPatchInstance(Ref<CartesianMeshPatch> v)
{
  m_amr_patches.add(v);
  m_amr_patches_pointer.add(v.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removeOnePatch(Integer index)
{
  m_available_group_index.add(m_amr_patches[index]->index());
  // info() << "_removeOnePatch() -- Save group_index : " << m_available_group_index.back();

  m_amr_patch_cell_groups_all[index].clear();
  m_amr_patch_cell_groups_all.remove(index);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    m_amr_patch_cell_groups_inpatch[index].clear();
    m_amr_patch_cell_groups_inpatch.remove(index);
    m_amr_patch_cell_groups_overlap[index].clear();
    m_amr_patch_cell_groups_overlap.remove(index);
  }

  m_amr_patches_pointer.remove(index);
  m_amr_patches.remove(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// The array must be sorted.
void CartesianPatchGroup::
_removeMultiplePatches(ConstArrayView<Integer> indexes)
{
  Integer count = 0;
  for (const Integer index : indexes) {
    _removeOnePatch(index - count);
    count++;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removeAllPatches()
{
  for (Integer i = 1; i < m_amr_patch_cell_groups_all.size(); ++i) {
    m_amr_patch_cell_groups_all[i].clear();
  }
  m_amr_patch_cell_groups_all.clear();

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    for (Integer i = 0; i < m_amr_patch_cell_groups_inpatch.size(); ++i) {
      m_amr_patch_cell_groups_inpatch[i].clear();
      m_amr_patch_cell_groups_overlap[i].clear();
    }
    m_amr_patch_cell_groups_inpatch.clear();
    m_amr_patch_cell_groups_overlap.clear();
  }

  m_amr_patches_pointer.clear();
  m_amr_patches.clear();
  m_available_group_index.clear();
  m_patches_to_delete.clear();
  m_index_new_patches = 1;

  m_higher_level = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_createGroundPatch()
{
  if (!m_amr_patches.empty())
    return;
  auto patch = makeRef(new CartesianMeshPatch(m_cmesh, -1));

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
    patch->_internalApi()->positionRef().setMinPoint({ 0, 0, 0 });
    patch->_internalApi()->positionRef().setMaxPoint({ numbering->globalNbCellsX(0), numbering->globalNbCellsY(0), numbering->globalNbCellsZ(0) });
    patch->_internalApi()->positionRef().setLevel(0);
  }

  _addPatchInstance(patch);
  _addCellGroup(m_cmesh->mesh()->allLevelCells(0), patch.get(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
_addCellGroup(CellGroup cell_group, CartesianMeshPatch* patch, bool add_flags)
{
  m_amr_patch_cell_groups_all.add(cell_group);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    // Irregular patch.
    // m_amr_patch_cell_groups_inpatch.add(cell_group);
    // m_amr_patch_cell_groups_overlap.add(CellGroup());
    return m_amr_patch_cell_groups_all.size() - 1;
  }

  AMRPatchPosition patch_position = patch->position();
  Ref<ICartesianMeshNumberingMngInternal> numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  UniqueArray<Int32> inpatch_items_lid;
  UniqueArray<Int32> overlap_items_lid;

  ENUMERATE_ (Cell, icell, cell_group) {
    Cell cell = *icell;
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(cell);

    if (patch_position.isIn(pos)) {
      inpatch_items_lid.add(cell.localId());
    }
    else {
      overlap_items_lid.add(cell.localId());
    }
  }

  CellGroup own = m_cmesh->mesh()->cellFamily()->createGroup(cell_group.name().clone() + "_InPatch", inpatch_items_lid, true);
  m_amr_patch_cell_groups_inpatch.add(own);

  CellGroup overlap = m_cmesh->mesh()->cellFamily()->createGroup(cell_group.name().clone() + "_Overlap", overlap_items_lid, true);
  m_amr_patch_cell_groups_overlap.add(overlap);

  if (add_flags) {
    // If an entity is in a patch, it gets the II_InPatch flag.
    // If an entity is an overlap entity for a patch, it gets the II_Overlap flag.
    // As its name suggests, an overlap entity can overlap another patch. Therefore, an entity can be both II_InPatch and II_Overlap.
    ENUMERATE_ (Cell, icell, own) {
      icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
    }
    ENUMERATE_ (Face, iface, own.faceGroup()) {
      iface->mutableItemBase().addFlags(ItemFlags::II_InPatch);
    }
    ENUMERATE_ (Node, inode, own.nodeGroup()) {
      inode->mutableItemBase().addFlags(ItemFlags::II_InPatch);
    }

    ENUMERATE_ (Cell, icell, overlap) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
    }
    ENUMERATE_ (Face, iface, overlap.faceGroup()) {
      iface->mutableItemBase().addFlags(ItemFlags::II_Overlap);
    }
    ENUMERATE_ (Node, inode, overlap.nodeGroup()) {
      inode->mutableItemBase().addFlags(ItemFlags::II_Overlap);
    }
  }

  return m_amr_patch_cell_groups_all.size() - 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_updateCellGroups(Integer index, bool update_flags)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }

  CellGroup patch_all_cells = allCells(index);
  CellGroup patch_inpatch = inPatchCells(index);
  CellGroup patch_overlap = overlapCells(index);

  patch_all_cells.clear();
  patch_inpatch.clear();
  patch_overlap.clear();

  const auto& position = patch(index)->_internalApi()->positionRef();

  Int32 level = position.level();

  UniqueArray<Int32> inpatch_items_lid;
  UniqueArray<Int32> overlap_items_lid;

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // We add the flags to the patch cells.
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);

    if (position.isIn(pos)) {
      inpatch_items_lid.add(icell.localId());
    }
    else if (position.isInWithOverlap(pos)) {
      overlap_items_lid.add(icell.localId());
    }
  }

  patch_all_cells.addItems(inpatch_items_lid, false);
  patch_all_cells.addItems(overlap_items_lid, false);

  patch_inpatch.addItems(inpatch_items_lid, false);
  patch_overlap.addItems(overlap_items_lid, false);

  if (update_flags) {
    ENUMERATE_ (Cell, icell, patch_inpatch) {
      icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
    }
    ENUMERATE_ (Face, iface, patch_inpatch.faceGroup()) {
      iface->mutableItemBase().addFlags(ItemFlags::II_InPatch);
    }
    ENUMERATE_ (Node, inode, patch_inpatch.nodeGroup()) {
      inode->mutableItemBase().addFlags(ItemFlags::II_InPatch);
    }

    ENUMERATE_ (Cell, icell, patch_overlap) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
    }
    ENUMERATE_ (Face, iface, patch_overlap.faceGroup()) {
      iface->mutableItemBase().addFlags(ItemFlags::II_Overlap);
    }
    ENUMERATE_ (Node, inode, patch_overlap.nodeGroup()) {
      inode->mutableItemBase().addFlags(ItemFlags::II_Overlap);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// It is necessary that the source patch and the part_to_remove patch are in contact for this method to work.
void CartesianPatchGroup::
_removePartOfPatch(Integer index_patch_to_edit, const AMRPatchPosition& part_to_remove)
{
  // info() << "Coarse Zone"
  //                             << " -- Min point : " << part_to_remove.minPoint()
  //                             << " -- Max point : " << part_to_remove.maxPoint()
  //                             << " -- Level : " << part_to_remove.level();

  // p1 is the patch part that must be removed from p0.
  // We therefore only have four cases to handle (knowing that p0 and p1 must be in contact in x and/or y and/or z).
  //
  // Case 1:
  // p0   |-----|
  // p1 |---------|
  // r = {-1, -1}
  //
  // Case 2:
  // p0   |-----|
  // p1       |-----|
  // r = {p1_min, -1}
  //
  // Case 3:
  // p0   |-----|
  // p1 |-----|
  // r = {-1, p1_max}
  //
  // Case 4:
  // p0   |-----|
  // p1    |---|
  // r = {p1_min, p1_max}
  auto cut_points_p0 = [](CartCoord p0_min, CartCoord p0_max, CartCoord p1_min, CartCoord p1_max) -> std::pair<CartCoord, CartCoord> {
    std::pair to_return{ -1, -1 };
    if (p1_min > p0_min && p1_min < p0_max) {
      to_return.first = p1_min;
    }
    if (p1_max > p0_min && p1_max < p0_max) {
      to_return.second = p1_max;
    }
    return to_return;
  };

  ICartesianMeshPatch* patch = m_amr_patches_pointer[index_patch_to_edit];
  AMRPatchPosition patch_position = patch->position();

  UniqueArray<AMRPatchPosition> new_patch_out;

  CartCoord3 min_point_of_patch_to_exclude(-1, -1, -1);

  // Patch cutting around the area to exclude.
  {
    UniqueArray<AMRPatchPosition> new_patch_in;

    // We cut the patch in x.
    {
      auto cut_point_x = cut_points_p0(patch_position.minPoint().x, patch_position.maxPoint().x, part_to_remove.minPoint().x, part_to_remove.maxPoint().x);

      // p0   |-----|
      // p1 |---------|
      if (cut_point_x.first == -1 && cut_point_x.second == -1) {
        min_point_of_patch_to_exclude.x = patch_position.minPoint().x;
        new_patch_out.add(patch_position);
      }
      // p0   |-----|
      // p1       |-----|
      else if (cut_point_x.second == -1) {
        min_point_of_patch_to_exclude.x = cut_point_x.first;
        auto [fst, snd] = patch_position.cut(cut_point_x.first, MD_DirX);
        new_patch_out.add(fst);
        new_patch_out.add(snd);
      }
      // p0   |-----|
      // p1 |-----|
      else if (cut_point_x.first == -1) {
        min_point_of_patch_to_exclude.x = patch_position.minPoint().x;
        auto [fst, snd] = patch_position.cut(cut_point_x.second, MD_DirX);
        new_patch_out.add(fst);
        new_patch_out.add(snd);
      }
      // p0   |-----|
      // p1    |---|
      else {
        min_point_of_patch_to_exclude.x = cut_point_x.first;
        auto [fst, snd_thr] = patch_position.cut(cut_point_x.first, MD_DirX);
        new_patch_out.add(fst);
        auto [snd, thr] = snd_thr.cut(cut_point_x.second, MD_DirX);
        new_patch_out.add(snd);
        new_patch_out.add(thr);
      }
    }

    // We cut the patch in y.
    {
      std::swap(new_patch_out, new_patch_in);

      auto cut_point_y = cut_points_p0(patch_position.minPoint().y, patch_position.maxPoint().y, part_to_remove.minPoint().y, part_to_remove.maxPoint().y);

      // p0   |-----|
      // p1 |---------|
      if (cut_point_y.first == -1 && cut_point_y.second == -1) {
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          min_point_of_patch_to_exclude.y = patch_x.minPoint().y;
          new_patch_out.add(patch_x);
        }
      }
      // p0   |-----|
      // p1       |-----|
      else if (cut_point_y.second == -1) {
        min_point_of_patch_to_exclude.y = cut_point_y.first;
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          auto [fst, snd] = patch_x.cut(cut_point_y.first, MD_DirY);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1 |-----|
      else if (cut_point_y.first == -1) {
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          min_point_of_patch_to_exclude.y = patch_x.minPoint().y;
          auto [fst, snd] = patch_x.cut(cut_point_y.second, MD_DirY);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1    |---|
      else {
        min_point_of_patch_to_exclude.y = cut_point_y.first;
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          auto [fst, snd_thr] = patch_x.cut(cut_point_y.first, MD_DirY);
          new_patch_out.add(fst);
          auto [snd, thr] = snd_thr.cut(cut_point_y.second, MD_DirY);
          new_patch_out.add(snd);
          new_patch_out.add(thr);
        }
      }
    }

    // We cut the patch in z.
    if (m_cmesh->mesh()->dimension() == 3) {
      std::swap(new_patch_out, new_patch_in);
      new_patch_out.clear();

      auto cut_point_z = cut_points_p0(patch_position.minPoint().z, patch_position.maxPoint().z, part_to_remove.minPoint().z, part_to_remove.maxPoint().z);

      // p0   |-----|
      // p1 |---------|
      if (cut_point_z.first == -1 && cut_point_z.second == -1) {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = patch_y.minPoint().z;
          new_patch_out.add(patch_y);
        }
      }
      // p0   |-----|
      // p1       |-----|
      else if (cut_point_z.second == -1) {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = cut_point_z.first;
          auto [fst, snd] = patch_y.cut(cut_point_z.first, MD_DirZ);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1 |-----|
      else if (cut_point_z.first == -1) {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = patch_y.minPoint().z;
          auto [fst, snd] = patch_y.cut(cut_point_z.second, MD_DirZ);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1    |---|
      else {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = cut_point_z.first;
          auto [fst, snd_thr] = patch_y.cut(cut_point_z.first, MD_DirZ);
          new_patch_out.add(fst);
          auto [snd, thr] = snd_thr.cut(cut_point_z.second, MD_DirZ);
          new_patch_out.add(snd);
          new_patch_out.add(thr);
        }
      }
    }
  }

  // Fusion and addition part.
  {
    if (m_cmesh->mesh()->dimension() == 2) {
      min_point_of_patch_to_exclude.z = 0;
    }
    // info() << "Nb of new patch before fusion : " << new_patch_out.size();
    // info() << "min_point_of_patch_to_exclude : " << min_point_of_patch_to_exclude;

    // We set to null the patch representing the patch part to be removed.
    for (AMRPatchPosition& new_patch : new_patch_out) {
      if (new_patch.minPoint() == min_point_of_patch_to_exclude) {
        new_patch.setLevel(-2); // Devient null.
      }
      // else {
      //   info() << "\tPatch before fusion"
      //                               << " -- Min point : " << new_patch.minPoint()
      //                               << " -- Max point : " << new_patch.maxPoint()
      //                               << " -- Level : " << new_patch.level();
      // }
    }

    AMRPatchPositionLevelGroup::fusionPatches(new_patch_out, false);

    // We add the new patches to the list of patches.
    // Integer d_nb_patch_final = 0;
    for (const auto& new_patch : new_patch_out) {
      if (!new_patch.isNull()) {
        // info() << "\tNew cut patch"
        //                                     << " -- Min point : " << new_patch.minPoint()
        //                                     << " -- Max point : " << new_patch.maxPoint()
        //                                     << " -- Level : " << new_patch.level();
        _addCutPatch(new_patch, allCells(index_patch_to_edit));
        // d_nb_patch_final++;
      }
    }
    // info() << "Nb of new patch after fusion : " << d_nb_patch_final;
  }

  removePatch(index_patch_to_edit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group)
{
  // If this method is used by another method than _removePartOfPatch(),
  // check if m_higher_level update is necessary.
  // (up until now, this is not useful since there will be a call to applyPatchEdit()).
  if (parent_patch_cell_group.null())
    ARCANE_FATAL("Null cell group");

  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  Integer group_index = _nextIndexForNewPatch();
  String patch_group_name = String("CartesianMeshPatchCells") + group_index;

  auto* cdi = new CartesianMeshPatch(m_cmesh, group_index, new_patch_position);
  _addPatchInstance(makeRef(cdi));

  UniqueArray<Int32> cells_local_id;

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
  ENUMERATE_ (Cell, icell, parent_patch_cell_group) {
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    if (new_patch_position.isIn(pos)) {
      cells_local_id.add(icell.localId());
    }
  }

  CellGroup parent_cells = cell_family->createGroup(patch_group_name, cells_local_id, true);
  // False car les flags sont mis à jour après.
  _addCellGroup(parent_cells, cdi, false);

  // info() << "_addCutPatch()"
  //                             << " -- m_amr_patch_cell_groups : " << m_amr_patch_cell_groups_all.size()
  //                             << " -- m_amr_patches : " << m_amr_patches.size()
  //                             << " -- group_index : " << group_index
  //                             << " -- cell_group name : " << m_amr_patch_cell_groups_all.back().name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
_addPatch(const AMRPatchPosition& new_patch_position)
{
  UniqueArray<Int32> cells_local_id;

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // We add the flags to the patch cells.
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(new_patch_position.level())) {
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);

    if (new_patch_position.isInWithOverlap(pos)) {
      cells_local_id.add(icell.localId());
    }
  }

  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  Integer group_index = _nextIndexForNewPatch();
  String patch_group_name = String("CartesianMeshPatchCells") + group_index;

  auto* cdi = new CartesianMeshPatch(m_cmesh, group_index, new_patch_position);

  _addPatchInstance(makeRef(cdi));
  CellGroup parent_cells = cell_family->createGroup(patch_group_name, cells_local_id, true);
  Integer array_index = _addCellGroup(parent_cells, cdi, true);

  // TODO: These two indices are really not a good idea...
  return array_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
