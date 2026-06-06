// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelReplicationInfo.h                                  (C) 2000-2025 */
/*                                                                           */
/* Information on subdomain replication.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELREPLICATION_H
#define ARCANE_CORE_IPARALLELREPLICATION_H
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
 * \brief Brief information on parallel subdomain replication.
 */
class ARCANE_CORE_EXPORT IParallelReplication
{
 public:

  virtual ~IParallelReplication() = default;

 public:

  //! Indicates if replication is active
  virtual bool hasReplication() const = 0;

  //! Number of replications
  virtual Int32 nbReplication() const = 0;

  //! Rank in the replication (from 0 to nbReplication()-1)
  virtual Int32 replicationRank() const = 0;

  /*!
   * \brief Indicates if this replication rank is the master.
   *
   * This is useful for example for outputs, so that only one
   * replica outputs the information.
   */
  virtual bool isMasterRank() const = 0;

  //! Rank in the master replication.
  virtual Int32 masterReplicationRank() const = 0;

  /*!
   * \brief Communicator associated with all replicas representing the same subdomain.
   *
   * Is 0 if there is no replication (hasReplication() is false).
   */
  virtual IParallelMng* replicaParallelMng() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
