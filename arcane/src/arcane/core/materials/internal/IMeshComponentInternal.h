// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshComponentInternal.h                                    (C) 2000-2024 */
/*                                                                           */
/* API interne Arcane de 'IMeshComponent'.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHCOMPONENTINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHCOMPONENTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne Arcane de 'IMeshComponent'.
 */
class ARCANE_CORE_EXPORT IMeshComponentInternal
{
 public:

  virtual ~IMeshComponentInternal() = default;

 public:

  //! Indexeur pour accéder aux variables partielles.
  virtual MeshMaterialVariableIndexer* variableIndexer() const =0;

  //! Vue sur les mailles du constituant.
  virtual ConstituentItemLocalIdListView constituentItemListView() const =0;

  //! Index pour accéder aux variables partielles.
  virtual Int32 variableIndexerIndex() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
