// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelReplication.h                                       (C) 2000-2020 */
/*                                                                           */
/* Information on subdomain replication.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_PARALLELREPLICATION_H
#define ARCANE_IMPL_PARALLELREPLICATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arcane/core/IParallelReplication.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Parallel
 * \brief Brief information on parallel subdomain replication.
 *
 * Replication consists of taking a set of parallelMng() and
 * duplicating this set several times, each set performing
 * in principle the same treatment unless explicit code dictates otherwise.
 * For example, it is possible to have a calculation that is performed
 * generally on 8 subdomains, and replicate this set 4 times. This
 * then uses the equivalent of 32 processes.
 *
 * This class contains the information on replication and is accessible
 * via IParallelMng::replication().
 */
class ARCANE_IMPL_EXPORT ParallelReplication
: public IParallelReplication
{
 public:

  //! Constructor without replication
  ParallelReplication();

  //! Constructor with replication
  ParallelReplication(Int32 rank, Int32 nb_rank, Ref<IParallelMng> replica_pm);
  virtual ~ParallelReplication();

 public:

  virtual bool hasReplication() const { return m_is_active; }
  virtual Int32 nbReplication() const { return m_nb_replication; }
  virtual Int32 replicationRank() const { return m_replication_rank; }
  virtual bool isMasterRank() const { return m_is_master_rank; }
  virtual Int32 masterReplicationRank() const { return m_master_replication_rank; }
  virtual IParallelMng* replicaParallelMng() const { return m_replica_parallel_mng.get(); }

 private:

  bool m_is_active;
  Int32 m_nb_replication;
  Int32 m_replication_rank;
  bool m_is_master_rank;
  Int32 m_master_replication_rank;
  Ref<IParallelMng> m_replica_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
