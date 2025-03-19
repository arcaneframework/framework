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
removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id, SharedArray<Integer> altered_patches)
{
  UniqueArray<Integer> size_of_patches_before(m_amr_patch_cell_groups.size());
  for (Integer i = 0; i < m_amr_patch_cell_groups.size(); ++i) {
    size_of_patches_before[i] = m_amr_patch_cell_groups[i].size();
  }
  m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, size_of_patches_before);

  for (CellGroup cells : m_amr_patch_cell_groups) {
    cells.removeItems(cells_local_id);
  }

  UniqueArray<Integer> size_of_patches_after(m_amr_patch_cell_groups.size());
  for (Integer i = 0; i < m_amr_patch_cell_groups.size(); ++i) {
    size_of_patches_after[i] = m_amr_patch_cell_groups[i].size();
  }
  m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, size_of_patches_after);

  altered_patches.clear();
  for (Integer i = 0; i < size_of_patches_after.size(); ++i) {
    if (size_of_patches_before[i] != size_of_patches_after[i]) {
      altered_patches.add(i+1);
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

// void CartesianPatchGroup::
// repairPatch(Integer index, ICartesianMeshNumberingMng* numbering_mng)
// {
//   UniqueArray<Integer> size_of_line;
//   UniqueArray<Integer> begin_of_line;
//
//   ENUMERATE_ ()
//     CellDirectionMng cells = m_amr_patches_pointer[index]->cellDirection(eMeshDirection::MD_DirX);
//   cells.cell()
//
//   // Localement, découper le old_patch en patch X.
//   // Echange pour avoir la taille réel des patchs X.
//   // Fusion des patchs X ayant le même originX et lenghtX (entre Y et Y+1).
//   // (3D) Fusion des patchs XY ayant le même originXY et lenghtXY (entre Z et Z+1).
// }

void CartesianPatchGroup::
updateLevelsBeforeCoarsen()
{
  for (ICartesianMeshPatch* patch : m_amr_patches_pointer) {
    patch->setLevel(patch->level() + 1);
  }
}

Integer CartesianPatchGroup::
nextIndexForNewPatch(){
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
