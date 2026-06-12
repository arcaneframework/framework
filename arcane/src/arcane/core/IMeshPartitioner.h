// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitioner.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh partitioner.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHPARTITIONER_H
#define ARCANE_IMESHPARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshPartitionerBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a mesh partitioner.
 *
 * The partitioner reassigns entity owners.
 * It does not directly perform entity exchange.
 * The partitioner can use certain information such as
 * timeRatio() or imbalance() to calculate an efficient partitioning.
 */
class ARCANE_CORE_EXPORT IMeshPartitioner
: public IMeshPartitionerBase
{
 public:

  virtual void build() = 0;

 public:

  using IMeshPartitionerBase::partitionMesh;

  virtual void partitionMesh(bool initial_partition, Int32 nb_part) = 0;

  //! Mesh associated with the partitioner
  ARCCORE_DEPRECATED_2021("Use primaryMesh() instead")
  virtual IMesh* mesh() const = 0;

  //! Associated mesh
  virtual IPrimaryMesh* primaryMesh() override;

 public:

  /*!{ \name compact
   *
   * Proportion of computation time of this subdomain compared to that
   * of the subdomain that has the highest computation time.
   */
  //! Sets the proportion of computation time
  //virtual void setTimeRatio(Real v) =0;
  //! Proportion of computation time
  //virtual Real timeRatio() const =0;
  //@}

  //! Computation time of the most heavily loaded subdomain
  virtual ARCANE_DEPRECATED_116 void setMaximumComputationTime(Real v) = 0;
  virtual ARCANE_DEPRECATED_116 Real maximumComputationTime() const = 0;

  /*! \brief Computation time of this subdomain.
   * The first element indicates the computation time of the subdomain
   * corresponding to calculations whose cost is proportional to the cells.
   * The following must be associated with a variable (to be done).
   */
  virtual ARCANE_DEPRECATED_116 void setComputationTimes(RealConstArrayView v) = 0;
  virtual ARCANE_DEPRECATED_116 RealConstArrayView computationTimes() const = 0;

  /*!@{ \name imbalance
   *
   * Computation time imbalance. It is calculated as follows
   * imbalance = (max_computation_time - min_computation_time) / min_computation_time;
   */
  //! Sets the computation time imbalance
  virtual void setImbalance(Real v) = 0;
  //! Computation time imbalance
  virtual Real imbalance() const = 0;
  //@}

  //! Sets the maximum allowed imbalance
  virtual void setMaxImbalance(Real v) = 0;
  //! Maximum allowed imbalance
  virtual Real maxImbalance() const = 0;

  //! Allows defining the weights of objects to be partitioned: ILoadBalanceMng must now be used.
  virtual ARCANE_DEPRECATED_116 void setCellsWeight(ArrayView<float> weights, Integer nb_weight) = 0;
  virtual ARCANE_DEPRECATED_116 ArrayView<float> cellsWeight() const = 0;

  //! Changes the ILoadBalanceMng to use.
  virtual void setILoadBalanceMng(ILoadBalanceMng* mng) = 0;
  virtual ILoadBalanceMng* loadBalanceMng() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
