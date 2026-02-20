// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineShMemWinBaseInternal.cc             (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour l'ensemble des      */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/* Les segments de ces fenêtres ne sont pas contigüs en mémoire et peuvent   */
/* être redimensionnés.                                                      */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/parallel/mpithread/internal/HybridMachineShMemWinBaseInternal.h"

#include "arcane/parallel/mpithread/HybridMessageQueue.h"

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/message_passing_mpi/internal/MpiMultiMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMachineShMemWinBaseInternal::
HybridMachineShMemWinBaseInternal(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<Mpi::MpiMultiMachineShMemWinBaseInternal> mpi_windows, IThreadBarrier* barrier)
: m_my_rank_local_proc(my_rank_local_proc)
, m_nb_rank_local_proc(nb_rank_local_proc)
, m_my_rank_mpi(my_rank_mpi)
, m_machine_ranks(ranks)
, m_sizeof_type(sizeof_type)
, m_mpi_windows(mpi_windows)
, m_thread_barrier(barrier)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HybridMachineShMemWinBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> HybridMachineShMemWinBaseInternal::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineShMemWinBaseInternal::
barrier() const
{
  m_thread_barrier->wait();
  if (m_my_rank_local_proc == 0)
    m_mpi_windows->barrier();
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> HybridMachineShMemWinBaseInternal::
segmentView()
{
  return m_mpi_windows->segmentView(m_my_rank_local_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> HybridMachineShMemWinBaseInternal::
segmentView(Int32 rank)
{
  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);
  const Int32 rank_local_proc = my_fri.localRankValue();
  const Int32 rank_mpi = my_fri.mpiRankValue();

  return m_mpi_windows->segmentView(rank_mpi, rank_local_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> HybridMachineShMemWinBaseInternal::
segmentConstView() const
{
  return m_mpi_windows->segmentConstView(m_my_rank_local_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> HybridMachineShMemWinBaseInternal::
segmentConstView(Int32 rank) const
{
  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);
  const Int32 rank_local_proc = my_fri.localRankValue();
  const Int32 rank_mpi = my_fri.mpiRankValue();

  return m_mpi_windows->segmentConstView(rank_mpi, rank_local_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineShMemWinBaseInternal::
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

void HybridMachineShMemWinBaseInternal::
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

void HybridMachineShMemWinBaseInternal::
addToAnotherSegment(Int32 rank, Span<const std::byte> elem)
{
  FullRankInfo fri = FullRankInfo::compute(MP::MessageRank(rank), m_nb_rank_local_proc);
  Int32 rank_local_proc = fri.localRankValue();
  Int32 rank_mpi = fri.mpiRankValue();

  m_mpi_windows->requestAddToAnotherSegment(m_my_rank_local_proc, rank_mpi, rank_local_proc, elem);
  m_thread_barrier->wait();

  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeAddToAnotherSegment();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineShMemWinBaseInternal::
addToAnotherSegment()
{
  m_thread_barrier->wait();

  if (m_my_rank_local_proc == 0) {
    m_mpi_windows->executeAddToAnotherSegment();
  }
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMachineShMemWinBaseInternal::
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

void HybridMachineShMemWinBaseInternal::
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

void HybridMachineShMemWinBaseInternal::
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

void HybridMachineShMemWinBaseInternal::
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

void HybridMachineShMemWinBaseInternal::
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
