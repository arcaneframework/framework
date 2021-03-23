// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelReplication.h                                       (C) 2000-2020 */
/*                                                                           */
/* Informations sur la réplication de sous-domaines.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_PARALLELREPLICATION_H
#define ARCANE_IMPL_PARALLELREPLICATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IParallelReplication.h"
#include "arcane/Parallel.h"
#include "arcane/utils/Ref.h"

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
 * \brief Informations sur la réplication des sous-domaines en parallèle.
 *
 * La réplication consiste à prendre un ensemble de parallelMng()et à
 * dupliquer cette ensemble plusieurs fois, chaque ensemble effectuant
 * à priori le même traitement sauf code explicite.
 * Par exemple, il est possible d'avoir un calcul qui s'effectue
 * en général sur 8 sous-domaines, et répliquer cet ensemble 4 fois. On
 * utilise alors l'équivalent de 32 processus.
 *
 * Cette classe contient les infos sur la réplication et est accessible
 * via IParallelMng::replication().
 */
class ARCANE_IMPL_EXPORT ParallelReplication
: public IParallelReplication
{
 public:

  //! Constructeur sans réplication
  ParallelReplication();
  //! Constructeur avec réplication
  ParallelReplication(Int32 rank,Int32 nb_rank,Ref<IParallelMng> replica_pm);
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
