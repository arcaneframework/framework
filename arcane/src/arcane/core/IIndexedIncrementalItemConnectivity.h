// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIndexedIncrementalItemConnectivity.h                       (C) 2000-2025 */
/*                                                                           */
/* Incremental connectivity interface for entities.                          */
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
 * \brief Interface to manage incremental connectivity.
 *
 * A connectivity links two families, a source (sourceFamily()) and
 * a target (targetFamily()).
 */
class ARCANE_CORE_EXPORT IIndexedIncrementalItemConnectivity
{
 public:

  virtual ~IIndexedIncrementalItemConnectivity() = default;

 public:

  //! Interface of the associated connectivity
  virtual IIncrementalItemConnectivity* connectivity() = 0;

  //! View of the connectivity.
  virtual IndexedItemConnectivityViewBase view() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
