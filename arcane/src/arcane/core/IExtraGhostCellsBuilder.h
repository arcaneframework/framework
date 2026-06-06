// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExtraGhostCellsBuilder.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface of a builder for "extraordinary" ghost cells                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IEXTRAGHOSTCELLSBUILDER_H
#define ARCANE_CORE_IEXTRAGHOSTCELLSBUILDER_H
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
 * \brief Interface of a builder for "extraordinary" ghost cells
 *
 * An "extraordinary" ghost cell is a ghost cell added to the
 * ghost cells defined by the mesh connectivity. Specifically,
 * the calculation of extraordinary ghost cells is performed at every update
 * of the mesh or load balancing.
 *
 * \note Makes the \a remove_old_ghost parameter of the IMesh::endUpdate() method obsolete.
 */
class IExtraGhostCellsBuilder
{
 public:

  virtual ~IExtraGhostCellsBuilder() {} //!< Frees resources.

 public:

  /*!
   * \brief Calculates the "extraordinary" cells to send.
   *
   * Performs the calculation of "extraordinary" cells following
   * a construction algorithm
   */
  virtual void computeExtraCellsToSend() = 0;

  /*!
   * \brief Local indices of "extraordinary" cells for sending.
   *
   * Retrieves the array of "extraordinary" cells destined
   * for subdomain \a sid
   */
  virtual Int32ConstArrayView extraCellsToSend(Int32 rank) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
