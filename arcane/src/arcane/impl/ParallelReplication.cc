// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelReplicationInfo.cc                                  (C) 2000-2020 */
/*                                                                           */
/* Informations sur la réplication de sous-domaines.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ParallelReplication.h"
#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelReplication::
ParallelReplication()
: m_is_active(false)
, m_nb_replication(1)
, m_replication_rank(0)
, m_is_master_rank(true)
, m_master_replication_rank(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelReplication::
ParallelReplication(Int32 replica_rank,Int32 nb_replica,Ref<IParallelMng> replica_pm)
: m_is_active(nb_replica!=1)
, m_nb_replication(nb_replica)
, m_replication_rank(replica_rank)
, m_is_master_rank(replica_rank==0)
, m_master_replication_rank(0)
, m_replica_parallel_mng(replica_pm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelReplication::
~ParallelReplication()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
