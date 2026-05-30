// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizerMpiCommunicator.h                      (C) 2000-2022 */
/*                                                                           */
/* Interface of a specific MPI communicator for synchronizations.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_IVARIABLESYNCHRONIZERMPICOMMUNICATOR_H
#define ARCANE_PARALLEL_MPI_IVARIABLESYNCHRONIZERMPICOMMUNICATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class VariableSynchronizer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface of a specific MPI communicator for synchronizations.
 *
 * This communicator allows the use of MPI 3.1 methods such as
 * MPI_Neighbor_alltoallv() for synchronizations.
 *
 * compute() must be called before this communicator can be used
 * specifically.
 */
class ARCANE_MPI_EXPORT IVariableSynchronizerMpiCommunicator
{
 public:

  virtual ~IVariableSynchronizerMpiCommunicator() = default;

 public:

  //! Calculates the specific communicator
  virtual void compute(VariableSynchronizer* var_syncer) = 0;

  /*!
   * \brief Retrieves the specific communicator from the topology.
   *
   * This communicator should not be retained because it may be invalidated
   * between two calls to compute().
   */
  virtual MPI_Comm communicator() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
