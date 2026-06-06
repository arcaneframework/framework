// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivity.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface for entity connectivity.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMCONNECTIVITY_H
#define ARCANE_CORE_IITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface to manage connectivity.
 *
 * A connectivity links two families, a source (sourceFamily()) and
 * a target (targetFamily()).
 *
 * To retrieve the target entities connected to a source entity, you must
 * use the ConnectivityItemVector class. For example:
 *
 * \code
 * IItemConnectivity* c = ...;
 * Item my_item;
 * ConnectivityItemVector civ(c);
 * ENUMERATE_ITEM(icitem,civ.connectedItems(my_item)){
 *  // Itère sur les entités connectées à \a my_item via \a c.
 * }
 * \endcode
 */
class ARCANE_CORE_EXPORT IItemConnectivity
: public IItemConnectivityAccessor
{
 public:

  friend class ConnectivityItemVector;

 public:

  //! Name of the connectivity
  virtual const String& name() const = 0;

  //! List of families (sourceFamily() + targetFamily())
  virtual ConstArrayView<IItemFamily*> families() const = 0;

  //! Source family
  virtual IItemFamily* sourceFamily() const = 0;

  //! Target family
  virtual IItemFamily* targetFamily() const = 0;

  //! Notifies the connectivity that the source family has been compacted.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) = 0;

  //! Notifies the connectivity that the target family has been compacted.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) = 0;

  /*!
   * \brief Update of the connectivity.
   *
   * Both arrays \a from_items and \a to_items are of the same size.
   * Items can therefore potentially be repeated if they appear in
   * multiple connections. For example, if we have the following connections (by lid) 1-3; 2-4; 1-5, we input
   * from_items [ 1 2 1 ] and to_items [ 3 4 5 ].
   */
  virtual void updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
