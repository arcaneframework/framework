// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizer.h                                  (C) 2000-2025 */
/*                                                                           */
/* Synchronization of the list of materials/media of entities.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZER_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/core/VariableTypedef.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/MatItem.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/materials/internal/IMeshMaterialSynchronizerImpl.h"
#include "arcane/materials/internal/AcceleratorMeshMaterialSynchronizerImpl.h"
#include "arcane/materials/internal/LegacyMeshMaterialSynchronizerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialModifierImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Synchronization of the list of materials/media of entities.
 *
 * This class allows synchronization between subdomains of the list
 * of materials/media to which a cell belongs.
 *
 * The ghost cells of this subdomain will retrieve the owned cells
 * and their list of materials/media. These ghost cells will then potentially
 * be added or removed from the current materials and media to be consistent
 * with this list derived from the owned cells.
 */
class MeshMaterialSynchronizer
: public TraceAccessor
{
 public:

  explicit MeshMaterialSynchronizer(IMeshMaterialMng* material_mng);
  ~MeshMaterialSynchronizer();

 public:

  /*!
   * \brief Synchronization of the list of materials/media of entities.
   *
   * This operation is collective.
   *
   * Returns \a true if cells have been added or removed from a material
   * or a medium during this operation for this subdomain.
   */
  bool synchronizeMaterialsInCells();
  void checkMaterialsInCells(Integer max_print);

 private:

  IMeshMaterialSynchronizerImpl* m_synchronizer;
  IMeshMaterialMng* m_material_mng;

  void _checkComponents(VariableCellInt32& indexes,
                        ConstArrayView<IMeshComponent*> components,
                        Integer max_print);
  void _checkComponentsInGhostCells(VariableCellInt64& hashes,
                                    Integer max_print);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
