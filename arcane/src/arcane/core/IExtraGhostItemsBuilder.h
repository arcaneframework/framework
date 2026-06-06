// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExtraGhostItemsBuilder.h                                   (C) 2000-2025 */
/*                                                                           */
/* Comment on file content.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IEXTRAGHOSTITEMSBUILDER_H_
#define ARCANE_CORE_IEXTRAGHOSTITEMSBUILDER_H_
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for an "extraordinary" ghost item builder
 *
 * An "extraordinary" ghost item is a ghost item added to the ghost items
 * defined by the mesh connectivity. Specifically, the calculation of
 * extraordinary ghost items is performed during every mesh update or load
 * balancing. This interface is particularly used for degrees of freedom.
 *
 * NOTE: makes the remove_old_ghost parameter of the IMesh::endUpdate method obsolete
 */
class ARCANE_CORE_EXPORT IExtraGhostItemsBuilder
{
 public:

  /** Class destructor */
  virtual ~IExtraGhostItemsBuilder() = default;

 public:

  /*!
   * \brief Calculation of "extraordinary" items to send
   * Performs the calculation of "extraordinary" items following
   * a construction algorithm
   */
  virtual void computeExtraItemsToSend() = 0;

  /*!
   * \brief Local indices of "extraordinary" items for sending
   * Retrieves the array of "extraordinary" items destined for
   * subdomain \a sid
   */
  virtual ConstArrayView<Int32> extraItemsToSend(Int32 sid) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
