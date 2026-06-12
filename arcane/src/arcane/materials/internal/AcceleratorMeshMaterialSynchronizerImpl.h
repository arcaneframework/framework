// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizer.h                                  (C) 2000-2024 */
/*                                                                           */
/* Synchronization of the list of materials/media of entities.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZERIMPLACC_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZERIMPLACC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/core/VariableTypedef.h"
#include "arcane/core/MeshVariableArrayRef.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/MatItem.h"

#include "arcane/accelerator/Accelerator.h"
#include "arcane/materials/internal/IndexSelecter.h"

#include "arcane/materials/internal/IMeshMaterialSynchronizerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialModifierImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Synchronization strategy for the list of materials/media of entities on the accelerator.
 *
 * This class allows synchronization between subdomains of the list
 * of materials/environment to which a cell belongs.
 *
 * The ghost cells of this subdomain will retrieve the list of materials/environment
 * from the owned cells. These ghost cells will then possibly have
 * materials and environment added or removed to be consistent
 * with this list from the owned cells.
 */
class AcceleratorMeshMaterialSynchronizerImpl
: public TraceAccessor
, public IMeshMaterialSynchronizerImpl
{
 public:

  explicit AcceleratorMeshMaterialSynchronizerImpl(IMeshMaterialMng* material_mng);

 public:

  /*!
   * \brief Synchronization of the list of materials/media of entities.
   *
   * This operation is collective.
   *
   * Returns true if cells were added or removed from a material
   * or a medium during this operation for this subdomain.
   */
  bool synchronizeMaterialsInCells();

 private:

  IMeshMaterialMng* m_material_mng;

 public:

  ARCCORE_HOST_DEVICE static void _setBit(Arcane::DataViewGetterSetter<unsigned char> bytes, Integer position)
  {
    Integer bit = position % 8;
    unsigned char temp = bytes;
    temp |= (Byte)(1 << bit);
    bytes = temp;
  }
  ARCCORE_HOST_DEVICE static bool _hasBit(Arcane::DataViewGetterSetter<unsigned char> bytes, Integer position)
  {
    Integer bit = position % 8;
    unsigned char temp = bytes;
    return temp & (1 << bit);
  }

 private:

  Arcane::Accelerator::IndexSelecter m_idx_selecter;
  VariableCellArrayByte m_mat_presence;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
