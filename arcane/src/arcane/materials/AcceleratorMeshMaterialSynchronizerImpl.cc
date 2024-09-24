// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorMeshMaterialSynchronizerImpl.cc                          (C) 2000-2023 */
/*                                                                           */
/* Synchronisation des entités des matériaux.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/AcceleratorMeshMaterialSynchronizerImpl.h"

#include "arcane/VariableTypes.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMesh.h"

#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialModifier.h"

#include "arcane/core/ItemGenericInfoListView.h"
#include "arcane/core/internal/IParallelMngInternal.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorMeshMaterialSynchronizerImpl::
AcceleratorMeshMaterialSynchronizerImpl(IMeshMaterialMng* material_mng)
: TraceAccessor(material_mng->traceMng())
, m_material_mng(material_mng)
, m_mat_presence(VariableBuildInfo(material_mng->mesh(), "ArcaneMaterialSyncPresence"))
{
  IMesh* mesh = m_material_mng->mesh();
  auto* internal_pm = mesh->parallelMng()->_internalApi();
  Arcane::Accelerator::RunQueue* m_queue = internal_pm->defaultQueue();
  m_idx_selecter = Arcane::Accelerator::IndexSelecter(m_queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorMeshMaterialSynchronizerImpl::
~AcceleratorMeshMaterialSynchronizerImpl()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void AcceleratorMeshMaterialSynchronizerImpl::
_setBit(Arcane::DataViewGetterSetter<unsigned char> bytes, Integer position)
{
  Integer bit = position % 8;
  unsigned char temp = bytes;
  temp |= (Byte)(1 << bit);
  bytes = temp;
}

inline bool AcceleratorMeshMaterialSynchronizerImpl::
_hasBit(Arcane::DataViewGetterSetter<unsigned char> bytes, Integer position)
{
  Integer bit = position % 8;
  unsigned char temp = bytes;
  return temp & (1 << bit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AcceleratorMeshMaterialSynchronizerImpl::
synchronizeMaterialsInCells()
{
  /*
    L'algorithme utilisé est le suivant:

    On utilise une variable aux mailles qui utilise un bit pour chaque
    matériau pour indiquer sa présence: si ce bit est positionné, le matériau
    est présent, sinon il est absent. La variable utilisée est donc de type
    ArrayByte aux mailles. Les méthodes _hasBit() et _setBit() permettent
    de positionner le bit d'un matériau donné.

    1. Le sous-domaine remplit cette variables pour ces mailles.
    2. La variable est synchronisée.
    3. Le sous-domaine compare pour chacune de ses mailles fantômes
    ce tableau de présence des matériaux et ajoute/supprime les matériaux en fonction
    de ce tableau.
  */
  IMesh* mesh = m_material_mng->mesh();
  if (!mesh->parallelMng()->isParallel())
    return false;

  auto* internal_pm = mesh->parallelMng()->_internalApi();

  ConstArrayView<IMeshMaterial*> materials = m_material_mng->materials();
  Integer nb_mat = materials.size();
  Integer dim2_size = nb_mat / 8;
  if ((nb_mat % 8) != 0)
    ++dim2_size;
  m_mat_presence.resize(dim2_size);

  info(4) << "Resize presence variable nb_mat=" << nb_mat << " dim2=" << dim2_size;
  CellToAllEnvCellConverter cell_converter = m_material_mng->cellToAllEnvCellConverter();

  Arcane::Accelerator::RunQueue* m_queue = internal_pm->defaultQueue();

  m_queue->setAsync(true);
  auto command = Arcane::Accelerator::makeCommand(m_queue);

  auto out_mat_presence = Arcane::Accelerator::viewInOut(command, m_mat_presence);
  CellToAllEnvCellAccessor cell2allenvcell(m_material_mng);

  m_idx_selecter.resize(mesh->allCells().size());

  command << RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL(cell2allenvcell, cid, mesh->ownCells())
  {
    AllEnvCell allenvcell{ cell_converter[cid] };

    for (Integer dim2 = 0; dim2 < dim2_size; dim2++) {
      out_mat_presence[cid][dim2] = 0;
    }

    ENUMERATE_CELL_ENVCELL (ienvcell, allenvcell) {
      ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
        MatCell mc = *imatcell;
        Integer mat_index = mc.materialId();
        _setBit(out_mat_presence[cid][mat_index / 8], mat_index);
      }
    }
  };

  bool has_changed = false;
  m_queue->barrier();

  m_mat_presence.synchronize();
  Arcane::ItemGenericInfoListView cells_info(mesh->cellFamily());
  auto out_after_presence = out_mat_presence;

  {
    UniqueArray<ConstArrayView<Int32>> to_add(nb_mat);
    UniqueArray<ConstArrayView<Int32>> to_remove(nb_mat);

    MeshMaterialModifier modifier(m_material_mng);

    for (Integer imat = 0; imat < nb_mat; ++imat) {

      to_add[imat] = m_idx_selecter.syncSelectIf(m_queue, [=] ARCCORE_HOST_DEVICE(Int32 cid) -> bool {
        if (cells_info.isOwn(cid))
          return false;
        CellLocalId c{ cid };
        AllEnvCell allenvcell{ cell_converter[c] };
        bool was_here = false;
        ENUMERATE_CELL_ENVCELL (ienvcell, allenvcell) {
          ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
            MatCell mc = *imatcell;
            Integer mat_index = mc.materialId();

            if (mat_index == imat) {
              was_here = true;
            }
          }
        }
        if (was_here)
          return false;
        return _hasBit(out_after_presence[c][imat / 8], imat);
      },
                                                 /*host_view=*/false);

      if (!to_add[imat].empty()) {
        /*String s="[";
        for(int u=0; u < to_add[imat].size(); u++){
          s = s + to_add[imat][u] + ",";
        }
        s = s +"]";
         pinfo() << "Materiau " << imat <<": " << to_add[imat].size() << " cellules ajoutées" << s << " adrr =" << to_add[imat];*/
        modifier.addCells(materials[imat], to_add[imat]);
        has_changed = true;
      }

      to_remove[imat] = m_idx_selecter.syncSelectIf(m_queue, [=] ARCCORE_HOST_DEVICE(Int32 cid) -> bool {
        if (cells_info.isOwn(cid))
          return false;
        CellLocalId c{ cid };
        AllEnvCell allenvcell{ cell_converter[c] };
        bool was_here = false;
        ENUMERATE_CELL_ENVCELL (ienvcell, allenvcell) {
          ENUMERATE_CELL_MATCELL (imatcell, (*ienvcell)) {
            MatCell mc = *imatcell;
            Integer mat_index = mc.materialId();

            if (mat_index == imat) {
              was_here = true;
            }
          }
        }
        if (!was_here)
          return false;
        return !_hasBit(out_after_presence[c][imat / 8], imat);
      },
                                                    /*host_view=*/false);

      if (!to_remove[imat].empty()) {
        /*String s="[";
        for(int u=0; u < to_remove[imat].size(); u++){
          s = s + to_remove[imat][u] + ",";
        }
        s = s +"]";
        pinfo() << "Materiau " << imat <<": " << to_remove[imat].size() << " cellules supprimées" << s;*/
        modifier.removeCells(materials[imat], to_remove[imat]);
        has_changed = true;
      }
    }
  }

  return has_changed;
}
} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
