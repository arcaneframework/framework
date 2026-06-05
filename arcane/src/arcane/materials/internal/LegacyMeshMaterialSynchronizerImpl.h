// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LegacyMeshMaterialSynchronizerImpl.h                        (C) 2000-2024 */
/*                                                                           */
/* Synchronization of the list of materials/media of entities.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZERIMPL_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/VariableTypedef.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/MatItem.h"

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
 * \brief Strategy for synchronizing the list of materials/media of entities.
 *
 * This class allows synchronizing the list
 * of materials/media to which a mesh belongs between subdomains.
 *
 * The ghost meshes of this subdomain will retrieve the meshes
 * of the proper meshes. These ghost meshes will then potentially
 * be added or removed from the current materials and media to be consistent
 * with this list derived from the proper meshes.
 */
class LegacyMeshMaterialSynchronizerImpl
: public TraceAccessor
, public IMeshMaterialSynchronizerImpl
{
 public:

  explicit LegacyMeshMaterialSynchronizerImpl(IMeshMaterialMng* material_mng);
  ~LegacyMeshMaterialSynchronizerImpl();

 public:

  /*!
   * \brief Synchronization of the list of materials/media of entities.
   *
   * This operation is collective.
   *
   * Returns \a true if meshes have been added or removed from a material
   * or a medium during this operation for this subdomain.
   */
  bool synchronizeMaterialsInCells();
  void checkMaterialsInCells(Integer max_print);

 private:

  IMeshMaterialMng* m_material_mng;

  inline static void _setBit(ByteArrayView bytes, Integer position);
  inline static bool _hasBit(ByteConstArrayView bytes, Integer position);
  void _fillPresence(AllEnvCell all_env_cell, ByteArrayView presence);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
