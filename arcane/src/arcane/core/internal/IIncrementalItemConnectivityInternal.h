// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIncrementalItemConnectivityInternal.h                      (C) 2000-2024 */
/*                                                                           */
/* API Arcane de l'interface de connectivité incrémentale des entités.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IINCREMENTALITEMCONNECTIVITYINTERNAL_H
#define ARCANE_CORE_INTERNAL_IINCREMENTALITEMCONNECTIVITYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer une connectivité incrémentale.
 *
 * Une connectivité relie deux familles, une source (sourceFamily()) et
 * une cible (targetFamily()).
 */
class ARCANE_CORE_EXPORT IIncrementalItemConnectivityInternal
{
 public:

  //TODO rendre 'protected' une fois que tout le monde utilisera le compteur de référence
  virtual ~IIncrementalItemConnectivityInternal() = default;

 public:

  virtual void shrinkMemory() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
