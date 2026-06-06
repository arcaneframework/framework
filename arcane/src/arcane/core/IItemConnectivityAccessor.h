// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityAccessor.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface for entity connectivity accessors.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMCONNECTIVITYACCESSOR_H
#define ARCANE_CORE_IITEMCONNECTIVITYACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface to manage access to a connectivity.
 */
class ARCANE_CORE_EXPORT IItemConnectivityAccessor
{
  friend ConnectivityItemVector;

 public:

  virtual ~IItemConnectivityAccessor() = default;

 public:

  //! Number of entities connected to the source entity with local ID \a lid
  virtual Integer nbConnectedItem(ItemLocalId lid) const = 0;

  //! localId() of the \a index-th entity connected to the source entity with local ID \a lid
  virtual Int32 connectedItemLocalId(ItemLocalId lid, Integer index) const = 0;

 protected:

  //! Implements the initialization of \a civ for this connectivity.
  virtual void _initializeStorage(ConnectivityItemVector* civ) = 0;

  //! Fills \a con_items with the entities connected to \a item.
  virtual ItemVectorView _connectedItems(ItemLocalId item, ConnectivityItemVector& con_items) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
