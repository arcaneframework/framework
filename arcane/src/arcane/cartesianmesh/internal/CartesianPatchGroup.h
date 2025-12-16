// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatchGroup.h                                       (C) 2000-2025 */
/*                                                                           */
/* Gestion du groupe de patchs du maillage cartésien.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANPATCHGROUP_H
#define ARCANE_CARTESIANMESH_CARTESIANPATCHGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/core/ItemGroup.h"

#include "arcane/utils/UniqueArray.h"

#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshPatch;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT CartesianPatchGroup
{
 public:

  explicit CartesianPatchGroup(ICartesianMesh* cmesh);

 public:

  Ref<CartesianMeshPatch> groundPatch();

  void addPatch(ConstArrayView<Int32> cells_local_id);
  Integer addPatchAfterRestore(CellGroup cell_group);
  void addPatch(CellGroup cell_group, Integer group_index);

  void addPatch(const AMRZonePosition& zone_position);

  Integer nbPatch() const;

  Ref<CartesianMeshPatch> patch(Integer index) const;

  CartesianMeshPatchListView patchListView() const;

  CellGroup allCells(Integer index);
  CellGroup inPatchCells(Integer index);
  CellGroup overallCells(Integer index);

  void clear();

  void removePatch(Integer index);

  void removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id);

  void removeCellsInAllPatches(const AMRPatchPosition& zone_to_delete);

  void applyPatchEdit(bool remove_empty_patches);

  void updateLevelsAndAddGroundPatch();

  void mergePatches();

  void refine(bool clear_refine_flag);

  void clearRefineRelatedFlags() const;

  void rebuildAvailableGroupIndex(ConstArrayView<Integer> available_group_index);

  ConstArrayView<Int32> availableGroupIndex();

  void setOverlapLayerSizeTopLevel(Integer size_of_overlap_layer_top_level);
  Integer overlapLayerSize(Integer level);

 private:

  Integer _nextIndexForNewPatch();

  void _addPatchInstance(Ref<CartesianMeshPatch> v);

  void _removeOnePatch(Integer index);
  void _removeMultiplePatches(ConstArrayView<Integer> indexes);
  void _removeAllPatches();
  void _createGroundPatch();

  void _addCellGroup(CellGroup cell_group, CartesianMeshPatch* patch);

  bool _isPatchInContact(const AMRPatchPosition& patch_position0, const AMRPatchPosition& patch_position1);
  void _splitPatch(Integer index_patch, const AMRPatchPosition& patch_position);
  void _addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group);
  void _addPatch(const AMRPatchPosition& new_patch_position);

 private:

  UniqueArray<CellGroup> m_amr_patch_cell_groups_all;
  UniqueArray<CellGroup> m_amr_patch_cell_groups_inpatch;
  UniqueArray<CellGroup> m_amr_patch_cell_groups_overall;
  UniqueArray<ICartesianMeshPatch*> m_amr_patches_pointer;
  UniqueArray<Ref<CartesianMeshPatch>> m_amr_patches;
  ICartesianMesh* m_cmesh;
  UniqueArray<Integer> m_patches_to_delete;
  Int32 m_index_new_patches;
  UniqueArray<Integer> m_available_group_index;
  Integer m_size_of_overlap_layer_sub_top_level;
  Integer m_higher_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
