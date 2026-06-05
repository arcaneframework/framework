// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizerImpl.cc                            (C) 2000-2024 */
/*                                                                           */
/* Synchronization of material entities.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/LegacyMeshMaterialSynchronizerImpl.h"

#include "arcane/VariableTypes.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMesh.h"

#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialModifier.h"

#include "arcane/core/ItemGenericInfoListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LegacyMeshMaterialSynchronizerImpl::
LegacyMeshMaterialSynchronizerImpl(IMeshMaterialMng* material_mng)
: TraceAccessor(material_mng->traceMng())
, m_material_mng(material_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LegacyMeshMaterialSynchronizerImpl::
~LegacyMeshMaterialSynchronizerImpl()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void LegacyMeshMaterialSynchronizerImpl::
_setBit(ByteArrayView bytes, Integer position)
{
  Integer offset = position / 8;
  Integer bit = position % 8;
  bytes[offset] |= (Byte)(1 << bit);
}

inline bool LegacyMeshMaterialSynchronizerImpl::
_hasBit(ByteConstArrayView bytes, Integer position)
{
  Integer offset = position / 8;
  Integer bit = position % 8;
  return bytes[offset] & (1 << bit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshMaterialSynchronizerImpl::
_fillPresence(AllEnvCell all_env_cell, ByteArrayView presence)
{
  ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
    ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
      MatCell mc = *imatcell;
      Integer mat_index = mc.materialId();
      _setBit(presence, mat_index);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool LegacyMeshMaterialSynchronizerImpl::
synchronizeMaterialsInCells()
{
  /*
    The algorithm used is as follows:

    We use a mesh variable that uses a bit for each material to indicate its
    presence: if this bit is set, the material is present; otherwise, it is
    absent. The variable used is therefore of type ArrayByte per mesh. The
    _hasBit() and _setBit() methods allow positioning the bit for a given
    material.

    1. The subdomain fills this variable for these meshes.
    2. The variable is synchronized.
    3. The subdomain compares this material presence table for each of its
    ghost meshes and adds/removes materials based on this table.
  */
  IMesh* mesh = m_material_mng->mesh();
  if (!mesh->parallelMng()->isParallel())
    return false;

  ConstArrayView<IMeshMaterial*> materials = m_material_mng->materials();
  Integer nb_mat = materials.size();
  VariableCellArrayByte mat_presence(VariableBuildInfo(mesh, "ArcaneMaterialSyncPresence"));
  Integer dim2_size = nb_mat / 8;
  if ((nb_mat % 8) != 0)
    ++dim2_size;
  mat_presence.resize(dim2_size);
  info(4) << "Resize presence variable nb_mat=" << nb_mat << " dim2=" << dim2_size;
  CellToAllEnvCellConverter cell_converter = m_material_mng->cellToAllEnvCellConverter();
  ENUMERATE_CELL (icell, mesh->ownCells()) {
    ByteArrayView presence = mat_presence[icell];
    presence.fill(0);
    AllEnvCell all_env_cell = cell_converter[*icell];
    _fillPresence(all_env_cell, presence);
  }

  bool has_changed = false;

  mat_presence.synchronize();
  {
    ByteUniqueArray before_presence(dim2_size);
    UniqueArray<UniqueArray<Int32>> to_add(nb_mat);
    UniqueArray<UniqueArray<Int32>> to_remove(nb_mat);
    ENUMERATE_CELL (icell, mesh->allCells()) {
      Cell cell = *icell;
      // Only processes ghost meshes.
      if (cell.isOwn())
        continue;
      Int32 cell_lid = cell.localId();
      AllEnvCell all_env_cell = cell_converter[cell];
      before_presence.fill(0);
      _fillPresence(all_env_cell, before_presence);
      ByteConstArrayView after_presence = mat_presence[cell];
      // Adds/Removes this mesh from materials if necessary.
      for (Integer imat = 0; imat < nb_mat; ++imat) {
        bool has_before = _hasBit(before_presence, imat);
        bool has_after = _hasBit(after_presence, imat);
        if (has_before && !has_after) {
          to_remove[imat].add(cell_lid);
        }
        else if (has_after && !has_before)
          to_add[imat].add(cell_lid);
      }
    }

    MeshMaterialModifier modifier(m_material_mng);
    for (Integer i = 0; i < nb_mat; ++i) {
      if (!to_add[i].empty()) {
        modifier.addCells(materials[i], to_add[i]);
        has_changed = true;
      }
      if (!to_remove[i].empty()) {
        modifier.removeCells(materials[i], to_remove[i]);
        has_changed = true;
      }
    }
  }
  return has_changed;
}

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
