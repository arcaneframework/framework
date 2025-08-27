// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIndexedIncrementalItemConnectivity.h                       (C) 2000-2025 */
/*                                                                           */
/* Interface de connectivité incrémentale des entités.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IINDEXEDINCREMENTALITEMCONNECTIVITY_H
#define ARCANE_CORE_IINDEXEDINCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IIncrementalItemConnectivity;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer une connectivité incrémentale.
 *
 * Une connectivité relie deux familles, une source (sourceFamily()) et
 * une cible (targetFamily()).
 */
class ARCANE_CORE_EXPORT IIndexedIncrementalItemConnectivity
{
 public:

  virtual ~IIndexedIncrementalItemConnectivity() = default;

 public:

  //! Interface de la connectivité associée
  virtual IIncrementalItemConnectivity* connectivity() =0;

  //! Vue sur la connectivité.
  virtual IndexedItemConnectivityViewBase view() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
