// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryDynamicMachineMemoryWindowBaseInternal.cc       (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée.                                        */
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/internal/SharedMemoryDynamicMachineMemoryWindowBaseInternal.h"

#include "arcane/utils/FatalErrorException.h"

#include "arccore/concurrency/IThreadBarrier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryDynamicMachineMemoryWindowBaseInternal::
SharedMemoryDynamicMachineMemoryWindowBaseInternal(Int32 my_rank, Int32 nb_rank, ConstArrayView<Int32> ranks, Int32 sizeof_type, UniqueArray<std::byte>* windows, Int32* owner_segments, IThreadBarrier* barrier)
: m_my_rank(my_rank)
, m_ranks(ranks)
, m_sizeof_type(sizeof_type)
, m_windows(windows)
, m_windows_span(windows, nb_rank)
, m_owner_segments(owner_segments)
, m_owner_segments_span(owner_segments, nb_rank)
, m_barrier(barrier)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryDynamicMachineMemoryWindowBaseInternal::
~SharedMemoryDynamicMachineMemoryWindowBaseInternal()
{
  m_barrier->wait();
  if (m_my_rank == 0) {
    delete[] m_windows;
    delete[] m_owner_segments;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 SharedMemoryDynamicMachineMemoryWindowBaseInternal::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> SharedMemoryDynamicMachineMemoryWindowBaseInternal::
machineRanks() const
{
  return m_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
barrier() const
{
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> SharedMemoryDynamicMachineMemoryWindowBaseInternal::
segment()
{
  return m_windows_span[m_owner_segments_span[m_my_rank]];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> SharedMemoryDynamicMachineMemoryWindowBaseInternal::
segment(Int32 rank)
{
  return m_windows_span[m_owner_segments_span[rank]];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 SharedMemoryDynamicMachineMemoryWindowBaseInternal::
segmentOwner() const
{
  return m_owner_segments_span[m_my_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 SharedMemoryDynamicMachineMemoryWindowBaseInternal::
segmentOwner(Int32 rank) const
{
  return m_owner_segments_span[rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
add(Span<const std::byte> elem)
{
  if (elem.size() % m_sizeof_type != 0) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }
  m_windows_span[m_owner_segments_span[m_my_rank]].addRange(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
exchangeSegmentWith(Int32 rank)
{
  const Int32 exchange_with = rank;

  if (exchange_with == m_my_rank) {
    m_barrier->wait();
    m_barrier->wait();
    return;
  }

  const Int32 segment_i_have = m_owner_segments_span[m_my_rank];
  const Int32 segment_i_want = m_owner_segments_span[exchange_with];

  m_barrier->wait();

  m_owner_segments_span[m_my_rank] = segment_i_want;

  m_barrier->wait();

  if (m_owner_segments_span[exchange_with] != segment_i_have) {
    ARCCORE_FATAL("Exchange from {0} to {1} is blocked : {1} would like segment {2}",
                  m_ranks[m_my_rank], rank, m_owner_segments_span[exchange_with]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
exchangeSegmentWith()
{
  m_barrier->wait();
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
resetExchanges()
{
  m_barrier->wait();
  m_owner_segments_span[m_my_rank] = m_my_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
reserve(Int64 new_capacity)
{
  m_windows_span[m_owner_segments_span[m_my_rank]].reserve(new_capacity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
resize(Int64 new_size)
{
  m_windows_span[m_owner_segments_span[m_my_rank]].resize(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
shrink()
{
  m_windows_span[m_owner_segments_span[m_my_rank]].shrink();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
