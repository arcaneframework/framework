// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatchGroup.h                                       (C) 2000-2023 */
/*                                                                           */
/* TODO                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANPATCHGROUP_H
#define ARCANE_CARTESIANMESH_CARTESIANPATCHGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arccore/trace/ITraceMng.h"

#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"

#include "arcane/cartesianmesh/internal/CartesianMeshPatch.h" // TODO Internal à gérer

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT CartesianPatchGroup
{
 public:

  explicit CartesianPatchGroup(ICartesianMesh* cmesh) : m_cmesh(cmesh){}

 public:
  void addPatch(CellGroup cell_group)
  {
    if (cell_group.null()) ARCANE_FATAL("Null cell group");
    auto* cdi = new CartesianMeshPatch(m_cmesh, nbPatch());
    m_amr_patch_cell_groups.add(cell_group);
    _addPatchInstance(makeRef(cdi));
    m_cmesh->traceMng()->info() << "m_amr_patch_cell_groups : " << m_amr_patch_cell_groups.size()
          << " -- m_amr_patches : " << m_amr_patches.size()
          << " -- m_amr_patches_pointer : " << m_amr_patches_pointer.size()
          << " -- cell_group name : " << m_amr_patch_cell_groups[nbPatch()-1].name()
          << " -- index : " << nbPatch()-1
    ;
  }

  Integer nbPatch() const
  {
    ARCANE_ASSERT((m_amr_patch_cell_groups.size()==m_amr_patches.size()), ("Pb size of array patches1"))
    ARCANE_ASSERT((m_amr_patches.size() == m_amr_patches_pointer.size()), ("Pb size of array patches2"))
    return m_amr_patch_cell_groups.size();
  }

  Ref<CartesianMeshPatch> patch(const Integer index) const
  {
    return m_amr_patches[index];
  }

  CartesianMeshPatchListView patchListView() const
  {
    return CartesianMeshPatchListView{m_amr_patches_pointer};
  }

  CellGroup cells(const Integer index)
  {
    return m_amr_patch_cell_groups[index];
  }

  void clear()
  {
    m_amr_patch_cell_groups.clear();
    m_amr_patches_pointer.clear();
    m_amr_patches.clear();
  }

  void removePatch(Integer index)
  {
    if (m_patches_to_delete.contains(index)) {
      return;
    }
    if (index < 0 || index >= m_amr_patch_cell_groups.size()) {
      ARCANE_FATAL("Invalid index");
    }

    m_patches_to_delete.add(index);
  }

  void removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id)
  {
    for (CellGroup cells : m_amr_patch_cell_groups) {
      if (cells.isAllItems())
        continue;
      cells.removeItems(cells_local_id);
    }
  }

  void removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id, UniqueArray<Integer>& altered_patches)
  {
    UniqueArray<Integer> size_of_patches_before(m_amr_patch_cell_groups.size());
    for (Integer i = 0; i < m_amr_patch_cell_groups.size(); ++i) {
      size_of_patches_before[i] = m_amr_patch_cell_groups[i].size();
    }
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, size_of_patches_before);

    for (CellGroup cells : m_amr_patch_cell_groups) {
      if (cells.isAllItems()) continue;
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
        altered_patches.add(i);
      }
    }
  }

  void applyPatchEdit(bool remove_empty_patches)
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
          m_patches_to_delete.add(i);
        }
      }
      _removeMultiplePatches(m_patches_to_delete);
      m_patches_to_delete.clear();
    }
  }

  // void repairPatch(Integer index, ICartesianMeshNumberingMng* numbering_mng)
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

  void updateLevelsBeforeCoarsen()
  {
    for (ICartesianMeshPatch* patch : m_amr_patches_pointer) {
      patch->setLevel(patch->level() + 1);
    }
  }

 private:
  void _addPatchInstance(const Ref<CartesianMeshPatch>& v)
  {
    m_amr_patches.add(v);
    m_amr_patches_pointer.add(v.get());
  }

  void _removeOnePatch(Integer index)
  {
    m_amr_patch_cell_groups.remove(index);
    m_amr_patches_pointer.remove(index);
    m_amr_patches.remove(index);
  }
  void _removeMultiplePatches(ConstArrayView<Integer> indexes)
  {
    Integer count = 0;
    for (const Integer index : indexes) {
      _removeOnePatch(index - count);
      count++;
    }
  }

 private:

  UniqueArray<CellGroup> m_amr_patch_cell_groups;
  UniqueArray<ICartesianMeshPatch*> m_amr_patches_pointer;
  UniqueArray<Ref<CartesianMeshPatch>> m_amr_patches;
  ICartesianMesh* m_cmesh;
  UniqueArray<Integer> m_patches_to_delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

