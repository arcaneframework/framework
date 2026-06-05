// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialSynchronizerImpl.h                             (C) 2000-2024 */
/*                                                                           */
/* Synchronization of the list of materials/media of entities.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_IMESHMATERIALSYNCHRONIZERIMPL_H
#define ARCANE_MATERIALS_INTERNAL_IMESHMATERIALSYNCHRONIZERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/VariableTypedef.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/MatItem.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Strategy for synchronizing the list of materials/media of entities.
 *
 * This abstract class determines the synchronization method between the sub-domains and the list
 * of materials/media to which a mesh belongs.
 *
 */
class IMeshMaterialSynchronizerImpl
{
 protected:

  explicit IMeshMaterialSynchronizerImpl() {};

 public:

  virtual ~IMeshMaterialSynchronizerImpl() {};

  /*!
   * \brief Synchronization of the list of materials/media of entities.
   *
   * This operation is collective.
   *
   * Returns true if meshes were added or removed from a material
   * or a medium during this operation for this sub-domain.
   */
  virtual bool synchronizeMaterialsInCells() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
