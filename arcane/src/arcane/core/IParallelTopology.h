// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelTopology.h                                        (C) 2000-20255 */
/*                                                                           */
/* Information on the computing core allocation topology.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELTOPOLOGY_H
#define ARCANE_CORE_IPARALLELTOPOLOGY_H
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
 * \ingroup Parallel
 * \brief Information on the computing core allocation topology.
 *
 * An instance of this class is linked to an IParallelMng.
 *
 * It allows knowing how the ranks of this IParallelMng are allocated
 * on the cluster and within the processes.
 *
 */
class ARCANE_CORE_EXPORT IParallelTopology
{
 public:

  virtual ~IParallelTopology() = default; //!< Frees resources.

 public:

  //! Associated parallelism manager
  virtual IParallelMng* parallelMng() const = 0;

  //! Indicates if this rank is the master rank for a machine (node)
  virtual bool isMasterMachine() const = 0;

  //! List of ranks that are on the same machine
  virtual Int32ConstArrayView machineRanks() const = 0;

  /*!
   * \brief Rank of this instance in the list of machines (nodes).
   *
   * This rank is between 0 and masterMachineRanks().size().
   */
  virtual Int32 machineRank() const = 0;

  /*!
   * \brief List of master ranks for each machine (node).
   *
   * This list is the same for all ranks.
   */
  virtual Int32ConstArrayView masterMachineRanks() const = 0;

  //! Indicates if this rank is the master within the ranks of this process.
  virtual bool isMasterProcess() const = 0;

  //! List of ranks that are in the same process (in multi-threading)
  virtual Int32ConstArrayView processRanks() const = 0;

  /*!
   * \brief Rank of this instance in the list of processes.
   *
   * This rank is between 0 and masterProcessRanks().size().
   */
  virtual Int32 processRank() const = 0;

  /*!
   * \brief List of master ranks for each process.
   *
   * This list is the same for all ranks.
   */
  virtual Int32ConstArrayView masterProcessRanks() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
