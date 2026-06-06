// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitionConstraint.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh partitioning constraint.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHPARTITIONCONSTRAINT_H
#define ARCANE_CORE_IMESHPARTITIONCONSTRAINT_H
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
 * \brief Interface of a mesh partitioning constraint.
 *
 * Instances of this interface are managed by an
 * IMeshPartitionConstraintMng.
 */
class IMeshPartitionConstraint
{
 public:

  virtual ~IMeshPartitionConstraint() = default; //!< Releases resources

 public:

  /*!
   * \brief Adds a set of constraints on the meshes.
   *
   * Adds to the \a linked_cells array a set of pairs of
   * uniqueId() of meshes that must remain in the same partition.
   * For example, if meshes 25 and 23 must remain connected, it
   * is enough to call:
   * \code
   * linked_cells.add(23);
   * linked_cells.add(25);
   * \endcode
   *
   * You must always add pairs of uniqueId(), potentially repeating
   * the meshes. For example, if one wishes
   * that meshes 35, 37, and 39, one must do as follows:
   * \code
   * linked_cells.add(35);
   * linked_cells.add(37);
   * linked_cells.add(35);
   * linked_cells.add(39);
   * \endcode
   * The \a linked_cells array must have a size that is a multiple of 2.
   * The \a linked owners array indicates for each pair which subdomain
   * it must belong to.
   *
   * TODO: Remove \a linked_owners
   *
   * \warning: each pair must start with the cell having the smallest uid.
   * The "owner" indicates for every couple that corresponds to the first cell.
   */
  virtual void addLinkedCells(Int64Array& linked_cells, Int32Array& linked_owners) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
