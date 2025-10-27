﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

#include "arcane/core/ItemTypes.h"
#include "arcane/cartesianmesh/ICartesianMeshPatch.h"
#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class CartesianMeshPatch;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT CartesianPatchGroup
{
 public:

  explicit CartesianPatchGroup(ICartesianMesh* cmesh);

 public:

  Ref<CartesianMeshPatch> groundPatch();

  void addPatch(CellGroup cell_group);

  Integer nbPatch() const;

  Ref<CartesianMeshPatch> patch(Integer index) const;

  CartesianMeshPatchListView patchListView() const;

  CellGroup cells(Integer index);

  void clear();

  void removePatch(Integer index);

  void removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id);

  void removeCellsInAllPatches(const AMRPatchPosition& zone_to_delete);

  void applyPatchEdit(bool remove_empty_patches);

  void updateLevelsBeforeAddGroundPatch();

  Integer nextIndexForNewPatch();

  void mergePatches();

 private:

  void _addPatchInstance(Ref<CartesianMeshPatch> v);

  void _removeOnePatch(Integer index);
  void _removeMultiplePatches(ConstArrayView<Integer> indexes);
  void _createGroundPatch();

  bool _isPatchInContact(const AMRPatchPosition& patch_position0, const AMRPatchPosition& patch_position1);
  void _splitPatch(Integer index_patch, const AMRPatchPosition& patch_position);
  void _addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group);

 private:

  UniqueArray<CellGroup> m_amr_patch_cell_groups;
  UniqueArray<ICartesianMeshPatch*> m_amr_patches_pointer;
  UniqueArray<Ref<CartesianMeshPatch>> m_amr_patches;
  ICartesianMesh* m_cmesh;
  UniqueArray<Integer> m_patches_to_delete;
  Int32 m_index_new_patches;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
