// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatchGroup.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Gestion du groupe de patchs du maillage cartésien.                        */
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianPatchGroup.h"

#include "arcane/cartesianmesh/internal/CartesianMeshPatch.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arccore/trace/ITraceMng.h"

#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"
#include "arcane/core/MeshKind.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/Vector3.h"
#include "internal/ICartesianMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<CartesianMeshPatch> CartesianPatchGroup::
groundPatch()
{
  _createGroundPatch();
  return patch(0);
}

void CartesianPatchGroup::
addPatch(CellGroup cell_group)
{
  _createGroundPatch();
  if (cell_group.null()) ARCANE_FATAL("Null cell group");
  auto* cdi = new CartesianMeshPatch(m_cmesh, nextIndexForNewPatch()+1); // +1 pour reproduire l'ancien comportement.
  m_amr_patch_cell_groups.add(cell_group);
  _addPatchInstance(makeRef(cdi));
  m_cmesh->traceMng()->info() << "m_amr_patch_cell_groups : " << m_amr_patch_cell_groups.size()
        << " -- m_amr_patches : " << m_amr_patches.size()
        << " -- m_amr_patches_pointer : " << m_amr_patches_pointer.size()
        << " -- cell_group name : " << m_amr_patch_cell_groups[nextIndexForNewPatch()-1].name()
        << " -- index : " << nextIndexForNewPatch()-1
  ;

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMng();
    FixedArray<Int64, 6> min_n_max;
    min_n_max[0] = INT64_MAX;
    min_n_max[1] = INT64_MAX;
    min_n_max[2] = INT64_MAX;
    min_n_max[3] = -1;
    min_n_max[4] = -1;
    min_n_max[5] = -1;
    ArrayView min(min_n_max.view().subView(0, 3));
    ArrayView max(min_n_max.view().subView(3, 3));
    Int64 nb_cells = 0;
    ENUMERATE_ (Cell, icell, cell_group) {
      if (icell->isOwn())
        nb_cells++;
      Int64 pos_x = numbering->cellUniqueIdToCoordX(*icell);
      if (pos_x < min[MD_DirX])
        min[MD_DirX] = pos_x;
      if (pos_x > max[MD_DirX])
        max[MD_DirX] = pos_x;

      Int64 pos_y = numbering->cellUniqueIdToCoordY(*icell);
      if (pos_y < min[MD_DirY])
        min[MD_DirY] = pos_y;
      if (pos_y > max[MD_DirY])
        max[MD_DirY] = pos_y;

      Int64 pos_z = numbering->cellUniqueIdToCoordZ(*icell);
      if (pos_z < min[MD_DirZ])
        min[MD_DirZ] = pos_z;
      if (pos_z > max[MD_DirZ])
        max[MD_DirZ] = pos_z;
    }
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMin, min);
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, max);
    nb_cells = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, nb_cells);

    max[MD_DirX] += 1;
    max[MD_DirY] += 1;
    max[MD_DirZ] += 1;

    {
      Int64 nb_cells_patch = (max[MD_DirX] - min[MD_DirX]) * (max[MD_DirY] - min[MD_DirY]) * (max[MD_DirZ] - min[MD_DirZ]);
      if (nb_cells != nb_cells_patch) {
        ARCANE_FATAL("Not regular patch");
      }
    }
    cdi->position().setMinPoint({ min[MD_DirX], min[MD_DirY], min[MD_DirZ] });
    cdi->position().setMaxPoint({ max[MD_DirX], max[MD_DirY], max[MD_DirZ] });
    // TODO : Pas de level, à ajouter !
  }
}

// Attention : avant _createGroundPatch() = 0, après _createGroundPatch(); = 1
Integer CartesianPatchGroup::
nbPatch() const
{
  return m_amr_patches.size();
}

Ref<CartesianMeshPatch> CartesianPatchGroup::
patch(const Integer index) const
{
  return m_amr_patches[index];
}

CartesianMeshPatchListView CartesianPatchGroup::
patchListView() const
{
  return CartesianMeshPatchListView{m_amr_patches_pointer};
}

CellGroup CartesianPatchGroup::
cells(const Integer index)
{
  if (index == 0) {
    ARCANE_FATAL("You cannot get cells of ground patch with this method");
  }
  return m_amr_patch_cell_groups[index-1];
}

// Attention : efface aussi le ground patch. Nécessaire de le récupérer après coup.
void CartesianPatchGroup::
clear()
{
  m_amr_patch_cell_groups.clear();
  m_amr_patches_pointer.clear();
  m_amr_patches.clear();
  _createGroundPatch();
}

void CartesianPatchGroup::
removePatch(const Integer index)
{
  if (m_patches_to_delete.contains(index)) {
    return;
  }
  if (index == 0) {
    ARCANE_FATAL("You cannot remove ground patch");
  }
  if (index < 1 || index >= m_amr_patch_cell_groups.size()) {
    ARCANE_FATAL("Invalid index");
  }

  m_patches_to_delete.add(index);
}

void CartesianPatchGroup::
removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id)
{
  for (CellGroup cells : m_amr_patch_cell_groups) {
    cells.removeItems(cells_local_id);
  }
}

void CartesianPatchGroup::
removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id, const AMRPatchPosition& patch_position)
{
  // Pas de foreach : _splitPatch() ajoute des patchs.
  // Attention si suppression de la suppression en deux étapes : _splitPatch() supprime aussi des patchs.
  for (Integer i = 0; i < m_amr_patches_pointer.size(); ++i) {
    ICartesianMeshPatch* patch = m_amr_patches_pointer[i];
    if (_isPatchInContact(patch->position(), patch_position)) {
      m_amr_patch_cell_groups[i - 1].removeItems(cells_local_id); // TODO : toujours utile ? Le patch sera supprimé.
      _splitPatch(i, patch_position);
    }
  }
}

void CartesianPatchGroup::
applyPatchEdit(bool remove_empty_patches)
{
  _removeMultiplePatches(m_patches_to_delete);
  m_patches_to_delete.clear();

  if (remove_empty_patches) {
    UniqueArray<Integer> size_of_patches(m_amr_patch_cell_groups.size());
    for (Integer i = 0; i < m_amr_patch_cell_groups.size(); ++i) {
      size_of_patches[i] = m_amr_patch_cell_groups[i].size();
    }
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, size_of_patches);
    for (Integer i = 0; i < size_of_patches.size(); ++i) {
      if (size_of_patches[i] == 0) {
        m_patches_to_delete.add(i+1);
      }
    }
    _removeMultiplePatches(m_patches_to_delete);
    m_patches_to_delete.clear();
  }
}

void CartesianPatchGroup::
updateLevelsBeforeCoarsen()
{
  for (ICartesianMeshPatch* patch : m_amr_patches_pointer) {
    patch->position().setLevel(patch->position().level() + 1);
  }
}

Integer CartesianPatchGroup::
nextIndexForNewPatch()
{
  return m_amr_patch_cell_groups.size();
}

void CartesianPatchGroup::
_addPatchInstance(Ref<CartesianMeshPatch> v)
{
  m_amr_patches.add(v);
  m_amr_patches_pointer.add(v.get());
}

void CartesianPatchGroup::
_removeOnePatch(Integer index)
{
  m_amr_patch_cell_groups.remove(index-1);
  m_amr_patches_pointer.remove(index);
  m_amr_patches.remove(index);
}
void CartesianPatchGroup::
_removeMultiplePatches(ConstArrayView<Integer> indexes)
{
  Integer count = 0;
  for (const Integer index : indexes) {
    _removeOnePatch(index - count);
    count++;
  }
}

void CartesianPatchGroup::
_createGroundPatch()
{
  if (!m_amr_patches.empty()) return;
  _addPatchInstance(makeRef(new CartesianMeshPatch(m_cmesh, -1)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CartesianPatchGroup::
_isPatchInContact(const AMRPatchPosition& patch_position0, const AMRPatchPosition& patch_position1)
{
  return (
  (patch_position0.level() == patch_position1.level()) &&
  (patch_position0.maxPoint().x > patch_position1.minPoint().x && patch_position1.maxPoint().x > patch_position0.minPoint().x) &&
  (patch_position0.maxPoint().y > patch_position1.minPoint().y && patch_position1.maxPoint().y > patch_position0.minPoint().y) &&
  (m_cmesh->mesh()->dimension() == 2 || (patch_position0.maxPoint().z > patch_position1.minPoint().z && patch_position1.maxPoint().z > patch_position0.minPoint().z)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_splitPatch(Integer index_patch, const AMRPatchPosition& part_to_remove)
{
  // Il est nécessaire que le patch source et le patch part_to_remove soient
  // en contact pour que cette méthode fonctionne.

  auto cut_point_p0 = [](Int64 p0_min, Int64 p0_max, Int64 p1_min, Int64 p1_max) -> std::pair<Int64, Int64> {
    std::pair<Int64, Int64> to_return{ -1, -1 };
    if (p1_min > p0_min && p1_min < p0_max) {
      to_return.first = p1_min;
    }
    if (p1_max > p0_min && p1_max < p0_max) {
      to_return.second = p1_max;
    }
    return to_return;
  };

  auto cut_patch = [](const AMRPatchPosition& patch, Int64 cut_point, Integer dim) -> std::pair<AMRPatchPosition, AMRPatchPosition> {
    auto patch_max_cut = patch.maxPoint();
    auto patch_min_cut = patch.minPoint();

    if (dim == MD_DirX) {
      patch_max_cut.x = cut_point;
      patch_min_cut.x = cut_point;
    }
    else if (dim == MD_DirY) {
      patch_max_cut.y = cut_point;
      patch_min_cut.y = cut_point;
    }
    else {
      patch_max_cut.z = cut_point;
      patch_min_cut.z = cut_point;
    }

    AMRPatchPosition p0;
    p0.setLevel(patch.level());
    p0.setMinPoint(patch.minPoint());
    p0.setMaxPoint(patch_max_cut);

    AMRPatchPosition p1;
    p1.setLevel(patch.level());
    p1.setMinPoint(patch_min_cut);
    p1.setMaxPoint(patch.maxPoint());

    return { p0, p1 };
  };

  ICartesianMeshPatch* patch = m_amr_patches_pointer[index_patch];
  AMRPatchPosition patch_position = patch->position();

  UniqueArray<AMRPatchPosition> new_patch_x;
  UniqueArray<AMRPatchPosition> new_patch_y;
  UniqueArray<AMRPatchPosition> new_patch_z;

  Int64 patch_to_exclude_x = -1;
  Int64 patch_to_exclude_y = -1;
  Int64 patch_to_exclude_z = -1;

  {
    std::pair<Int64, Int64> cut_point_x = cut_point_p0(patch_position.minPoint().x, patch_position.maxPoint().x, part_to_remove.minPoint().x, part_to_remove.maxPoint().x);

    // p0   |-----|
    // p1 |---------|
    if (cut_point_x.first == -1 && cut_point_x.second == -1) {
      patch_to_exclude_x = patch_position.minPoint().x;
      new_patch_x.add(patch_position);
    }
    // p0   |-----|
    // p1       |-----|
    else if (cut_point_x.second == -1) {
      patch_to_exclude_x = cut_point_x.first;
      auto new_patch = cut_patch(patch_position, cut_point_x.first, MD_DirX);
      new_patch_x.add(new_patch.first);
      new_patch_x.add(new_patch.second);
    }
    // p0   |-----|
    // p1 |-----|
    else if (cut_point_x.first == -1) {
      patch_to_exclude_x = patch_position.minPoint().x;
      auto new_patch = cut_patch(patch_position, cut_point_x.second, MD_DirX);
      new_patch_x.add(new_patch.first);
      new_patch_x.add(new_patch.second);
    }
    // p0   |-----|
    // p1    |---|
    else {
      patch_to_exclude_x = cut_point_x.first;
      auto new_patch_0 = cut_patch(patch_position, cut_point_x.first, MD_DirX);
      new_patch_x.add(new_patch_0.first);
      auto new_patch_1 = cut_patch(new_patch_0.second, cut_point_x.second, MD_DirX);
      new_patch_x.add(new_patch_1.first);
      new_patch_x.add(new_patch_1.second);
    }
  }
  {
    std::pair<Int64, Int64> cut_point_y = cut_point_p0(patch_position.minPoint().y, patch_position.maxPoint().y, part_to_remove.minPoint().y, part_to_remove.maxPoint().y);

    // p0   |-----|
    // p1 |---------|
    if (cut_point_y.first == -1 && cut_point_y.second == -1) {
      for (auto patch_x : new_patch_x) {
        patch_to_exclude_y = patch_x.minPoint().y;
        new_patch_y.add(patch_x);
      }
    }
    // p0   |-----|
    // p1       |-----|
    else if (cut_point_y.second == -1) {
      for (auto patch_x : new_patch_x) {
        patch_to_exclude_y = cut_point_y.first;
        auto new_patch = cut_patch(patch_x, cut_point_y.first, MD_DirY);
        new_patch_y.add(new_patch.first);
        new_patch_y.add(new_patch.second);
      }
    }
    // p0   |-----|
    // p1 |-----|
    else if (cut_point_y.first == -1) {
      for (auto patch_x : new_patch_x) {
        patch_to_exclude_y = patch_x.minPoint().y;
        auto new_patch = cut_patch(patch_x, cut_point_y.second, MD_DirY);
        new_patch_y.add(new_patch.first);
        new_patch_y.add(new_patch.second);
      }
    }
    // p0   |-----|
    // p1    |---|
    else {
      for (auto patch_x : new_patch_x) {
        patch_to_exclude_y = cut_point_y.first;
        auto new_patch_0 = cut_patch(patch_x, cut_point_y.first, MD_DirY);
        new_patch_y.add(new_patch_0.first);
        auto new_patch_1 = cut_patch(new_patch_0.second, cut_point_y.second, MD_DirY);
        new_patch_y.add(new_patch_1.first);
        new_patch_y.add(new_patch_1.second);
      }
    }
  }
  if (m_cmesh->mesh()->dimension() == 3) {
    std::pair<Int64, Int64> cut_point_z = cut_point_p0(patch_position.minPoint().z, patch_position.maxPoint().z, part_to_remove.minPoint().z, part_to_remove.maxPoint().z);

    // p0   |-----|
    // p1 |---------|
    if (cut_point_z.first == -1 && cut_point_z.second == -1) {
      for (auto patch_y : new_patch_y) {
        patch_to_exclude_z = patch_y.minPoint().z;
        new_patch_z.add(patch_y);
      }
    }
    // p0   |-----|
    // p1       |-----|
    else if (cut_point_z.second == -1) {
      for (auto patch_y : new_patch_y) {
        patch_to_exclude_z = cut_point_z.first;
        auto new_patch = cut_patch(patch_y, cut_point_z.first, MD_DirZ);
        new_patch_z.add(new_patch.first);
        new_patch_z.add(new_patch.second);
      }
    }
    // p0   |-----|
    // p1 |-----|
    else if (cut_point_z.first == -1) {
      for (auto patch_y : new_patch_y) {
        patch_to_exclude_z = patch_y.minPoint().z;
        auto new_patch = cut_patch(patch_y, cut_point_z.second, MD_DirZ);
        new_patch_z.add(new_patch.first);
        new_patch_z.add(new_patch.second);
      }
    }
    // p0   |-----|
    // p1    |---|
    else {
      for (auto patch_y : new_patch_y) {
        patch_to_exclude_z = cut_point_z.first;
        auto new_patch_0 = cut_patch(patch_y, cut_point_z.first, MD_DirZ);
        new_patch_z.add(new_patch_0.first);
        auto new_patch_1 = cut_patch(new_patch_0.second, cut_point_z.second, MD_DirZ);
        new_patch_z.add(new_patch_1.first);
        new_patch_z.add(new_patch_1.second);
      }
    }
  }

  // TODO Voir pour fusion des AMRPatchPosition (ici ou après ?)

  if (m_cmesh->mesh()->dimension() == 2) {
    for (auto new_patch : new_patch_y) {
      if (new_patch.minPoint().x != patch_to_exclude_x && new_patch.minPoint().y != patch_to_exclude_y) {
        _addCutPatch(new_patch, m_amr_patch_cell_groups[index_patch - 1]);
      }
    }
  }
  else {
    for (auto new_patch : new_patch_z) {
      if (new_patch.minPoint().x != patch_to_exclude_x && new_patch.minPoint().y != patch_to_exclude_y && new_patch.minPoint().z != patch_to_exclude_z) {
        _addCutPatch(new_patch, m_amr_patch_cell_groups[index_patch - 1]);
      }
    }
  }
  removePatch(index_patch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group)
{
  if (parent_patch_cell_group.null())
    ARCANE_FATAL("Null cell group");

  // TODO : Le nextIndexForNewPatch() actuel ne fonctionnera pas si on s'amuse à créer, détruire et recréer des patchs !
  // TODO : Attribut que l'on incrémente de 1 lors de la création d'un patch.
  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  Integer index = nextIndexForNewPatch();
  String parent_group_name = String("CartesianMeshPatchParentCells") + index;

  auto* cdi = new CartesianMeshPatch(m_cmesh, index + 1);

  _addPatchInstance(makeRef(cdi));

  UniqueArray<Int32> cells_local_id;

  cdi->position().setLevel(new_patch_position.level());
  cdi->position().setMinPoint(new_patch_position.minPoint());
  cdi->position().setMaxPoint(new_patch_position.maxPoint());

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMng();
  ENUMERATE_ (Cell, icell, parent_patch_cell_group) {
    Int64 pos_x = numbering->cellUniqueIdToCoordX(*icell);
    Int64 pos_y = numbering->cellUniqueIdToCoordY(*icell);
    Int64 pos_z = numbering->cellUniqueIdToCoordZ(*icell);
    if (new_patch_position.isIn(pos_x, pos_y, pos_z)) {
      cells_local_id.add(icell.localId());
    }
  }

  CellGroup parent_cells = cell_family->createGroup(parent_group_name, cells_local_id, true);
  m_amr_patch_cell_groups.add(parent_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
