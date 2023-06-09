// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyInternal.h                                       (C) 2000-2023 */
/*                                                                           */
/* Partie interne à Arcane de IItemFamily.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IITEMFAMILYINTERNAL_H
#define ARCANE_CORE_INTERNAL_IITEMFAMILYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamilyTopologyModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne de IItemFamily.
 */
class ARCANE_CORE_EXPORT IItemFamilyInternal
{
 public:

  virtual ~IItemFamilyInternal() = default;

 public:

  //! Informations sur les connectivités non structurés
  virtual ItemInternalConnectivityList* unstructuredItemInternalConnectivityList() = 0;

  //! Interface du modificateur de topologie.
  virtual IItemFamilyTopologyModifier* topologyModifier() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
