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

#include "CartesianMeshPatchListView.h"
#include "ICartesianMesh.h"
#include "arcane/core/ItemTypes.h"
#include "arccore/trace/ITraceMng.h"
#include "internal/CartesianMeshPatch.h" // TODO A gérer

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
    m_amr_patch_cell_groups.remove(index);
    m_amr_patches_pointer.remove(index);
    m_amr_patches.remove(index);
  }

  void removeCellsInPatch(Integer index, ConstArrayView<Int32> parent_cells_local_id)
  {

  }

 private:
  void _addPatchInstance(const Ref<CartesianMeshPatch>& v)
  {
    m_amr_patches.add(v);
    m_amr_patches_pointer.add(v.get());
  }
 private:

  UniqueArray<CellGroup> m_amr_patch_cell_groups;
  UniqueArray<ICartesianMeshPatch*> m_amr_patches_pointer;
  UniqueArray<Ref<CartesianMeshPatch>> m_amr_patches;
  ICartesianMesh* m_cmesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

