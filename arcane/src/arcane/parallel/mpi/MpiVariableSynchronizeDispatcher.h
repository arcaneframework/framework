// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiVariableSynchronizeDispatcher.h                          (C) 2000-2021 */
/*                                                                           */
/* Gestion spécifique MPI des synchronisations des variables.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIVARIABLESYNCHRONIZEDISPATCHER_H
#define ARCANE_PARALLEL_MPI_MPIVARIABLESYNCHRONIZEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/ParallelMngDispatcher.h"

#include "arcane/parallel/mpi/ArcaneMpi.h"

#include "arcane/impl/VariableSynchronizer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiVariableSynchronizeDispatcherBuildInfo
{
 public:
  MpiVariableSynchronizeDispatcherBuildInfo(MpiParallelMng* pm, GroupIndexTable* table)
  : m_parallel_mng(pm), m_table(table) { }
 public:
  MpiParallelMng* parallelMng() const{ return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
 private:
  MpiParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation optimisée pour MPI de la synchronisation.
 *
 * Par rapport à la version de base, cette implémentation fait un MPI_Waitsome
 * (au lieu d'un Waitall) et recopie dans le buffer de destination
 * dès qu'un message arrive.
 *
 * NOTE: cette optimisation respecte la norme MPI qui dit qu'on ne doit
 * plus toucher à la zone mémoire d'un message tant que celui-ci n'est
 * pas fini.
 */
template<typename SimpleType>
class MpiVariableSynchronizeDispatcher
: public VariableSynchronizeDispatcher<SimpleType>
{
 public:
  typedef typename VariableSynchronizeDispatcher<SimpleType>::SyncBuffer SyncBuffer;
 public:

  explicit MpiVariableSynchronizeDispatcher(MpiVariableSynchronizeDispatcherBuildInfo& bi);

  void compute(ItemGroupSynchronizeInfo* sync_list) override;
  void beginSynchronize(SyncBuffer& sync_buffer) override;
  void endSynchronize(SyncBuffer& sync_buffer) override;

 private:
  MpiParallelMng* m_mpi_parallel_mng;
  UniqueArray<Parallel::Request> m_original_recv_requests;
  UniqueArray<bool> m_original_recv_requests_done;
  Ref<Parallel::IRequestList> m_receive_request_list;
  Ref<Parallel::IRequestList> m_send_request_list;
  bool m_is_in_sync = false;
 private:
  void _copyReceive(SyncBuffer& sync_buffer,Integer index);
  void _copySend(SyncBuffer& sync_buffer,Integer index);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

