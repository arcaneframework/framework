// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitionConstraintMng.h                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh partitioning constraint manager.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHPARTITIONCONSTRAINTMNG_H
#define ARCANE_CORE_IMESHPARTITIONCONSTRAINTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a mesh partitioning constraint manager.
 */
class IMeshPartitionConstraintMng
{
 public:

  virtual ~IMeshPartitionConstraintMng() = default; //!< Releases resources

 public:

  //! Adds a constraint
  virtual void addConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Removes a constraint.
   *
   * The caller becomes the owner of \a constraint and must
   * manage its destruction.
   */
  virtual void removeConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Computes and applies constraints.
   *
   * It computes constraints on the mesh by applying
   * the IMeshPartitionConstraint::addLinkedCell() method for each constraint
   * and applies them by modifying the IItemFamily::itemsNewOwner() variable
   * of the cells. All cells that must be linked in it are then placed
   * in the same partition.
   *
   * This operation does not guarantee that the resulting partitions are
   * balanced in terms of load. For this,
   * a re-partitioning service (IMeshPartitioner) must be used
   * that takes these constraints into account.
   *
   * This operation is collective.
   */
  virtual void computeAndApplyConstraints() = 0;

  /*!
   * \brief Computes constraints and returns a list of linked entities.
   *
   * It computes constraints like computeAndApplyConstraints()
   * but does not modify the owner. Instead, it returns a
   * list containing lists of the uniqueId() of the entities that must
   * be linked.
   *
   * This operation is collective.
   */
  virtual void computeConstraintList(Int64MultiArray2& tied_uids) = 0;

  //! Adds a weak constraint
  virtual void addWeakConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Removes a constraint.
   *
   * The caller becomes the owner of \a constraint and must
   * manage its destruction.
   */
  virtual void removeWeakConstraint(IMeshPartitionConstraint* constraint) = 0;

  /*!
   * \brief Computes and applies constraints.
   *
   * It computes constraints on the mesh by applying
   * the IMeshPartitionConstraint::addLinkedCell() method for each constraint
   * and applies them by modifying the IItemFamily::itemsNewOwner() variable
   * of the cells. All cells that must be linked in it are then placed
   * in the same partition.
   *
   * This operation does not guarantee that the resulting partitions are
   * balanced in terms of load. For this,
   * a re-partitioning service (IMeshPartitioner) must be used
   * that takes these constraints into account.
   *
   * This operation is collective.
   */
  virtual void computeAndApplyWeakConstraints() = 0;

  /*!
   * \brief Computes constraints and returns a list of linked entities.
   *
   * It computes constraints like computeAndApplyConstraints()
   * but does not modify the owner. Instead, it returns a
   * list containing lists of the uniqueId() of the entities that must
   * be linked.
   *
   * This operation is collective.
   */
  virtual void computeWeakConstraintList(Int64MultiArray2& tied_uids) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
