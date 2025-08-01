// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridDynamicMachineMemoryWindowBaseInternal.cc             (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/parallel/mpithread/internal/HybridDynamicMachineMemoryWindowBaseInternal.h"

#include "arcane/parallel/mpithread/HybridMessageQueue.h"

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/message_passing_mpi/internal/MpiDynamicMultiMachineMemoryWindowBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridDynamicMachineMemoryWindowBaseInternal::
HybridDynamicMachineMemoryWindowBaseInternal(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type,
                                             Mpi::MpiDynamicMultiMachineMemoryWindowBaseInternal* mpi_windows, IThreadBarrier* barrier)
: m_my_rank_local_proc(my_rank_local_proc)
, m_nb_rank_local_proc(nb_rank_local_proc)
, m_my_rank_mpi(my_rank_mpi)
, m_machine_ranks(ranks)
, m_sizeof_type(sizeof_type)
, m_mpi_windows(mpi_windows)
, m_thread_barrier(barrier)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridDynamicMachineMemoryWindowBaseInternal::
~HybridDynamicMachineMemoryWindowBaseInternal()
{
  if (m_my_rank_local_proc == 0) {
    delete m_mpi_windows;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HybridDynamicMachineMemoryWindowBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> HybridDynamicMachineMemoryWindowBaseInternal::
machineRanks() const
{
  return m_mpi_windows->machineRanks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
barrier() const
{
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0)
    m_mpi_windows->barrier();
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> HybridDynamicMachineMemoryWindowBaseInternal::
segment()
{
  return m_mpi_windows->segment(m_my_rank_local_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> HybridDynamicMachineMemoryWindowBaseInternal::
segment(Int32 rank)
{
  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);
  const Int32 rank_local_proc = my_fri.localRankValue();
  const Int32 rank_mpi = my_fri.mpiRankValue();

  return m_mpi_windows->segment(rank_mpi, rank_local_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HybridDynamicMachineMemoryWindowBaseInternal::
segmentOwner() const
{
  return m_mpi_windows->segmentPos(m_my_rank_local_proc) + m_mpi_windows->segmentOwner(m_my_rank_local_proc) * m_nb_rank_local_proc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HybridDynamicMachineMemoryWindowBaseInternal::
segmentOwner(Int32 rank) const
{
  FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);
  Int32 rank_local_proc = fri.localRankValue();
  Int32 rank_mpi = fri.mpiRankValue();

  return m_mpi_windows->segmentPos(rank_mpi, rank_local_proc) + m_mpi_windows->segmentOwner(rank_mpi, rank_local_proc) * m_nb_rank_local_proc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
add(Span<const std::byte> elem)
{
  m_mpi_windows->requestAdd(m_my_rank_local_proc, elem);
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeAdd();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
add()
{
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeAdd();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
exchangeSegmentWith(Int32 rank)
{
  FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);
  Int32 rank_local_proc = fri.localRankValue();
  Int32 rank_mpi = fri.mpiRankValue();

  m_mpi_windows->requestExchangeSegmentWith(m_my_rank_local_proc, rank_mpi, rank_local_proc);
  m_thread_barrier->wait();

  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeExchangeSegmentWith();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
exchangeSegmentWith()
{
  m_thread_barrier->wait();

  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeExchangeSegmentWith();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
resetExchanges()
{
  m_mpi_windows->resetExchanges();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
reserve(Int64 new_capacity)
{
  m_mpi_windows->requestReserve(m_my_rank_local_proc, new_capacity);
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeReserve();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
reserve()
{
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeReserve();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
resize(Int64 new_size)
{
  m_mpi_windows->requestResize(m_my_rank_local_proc, new_size);
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeResize();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
resize()
{
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeResize();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridDynamicMachineMemoryWindowBaseInternal::
shrink()
{
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeShrink();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
