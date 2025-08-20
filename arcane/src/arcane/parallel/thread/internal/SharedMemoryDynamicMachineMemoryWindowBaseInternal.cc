// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryDynamicMachineMemoryWindowBaseInternal.cc       (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour l'ensemble des      */
/* sous-domaines en mémoire partagée.                                        */
/* Les segments de ces fenêtres ne sont pas contigüs en mémoire et peuvent   */
/* être redimensionnés.                                                      */
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
SharedMemoryDynamicMachineMemoryWindowBaseInternal(Int32 my_rank, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<UniqueArray<UniqueArray<std::byte>>> windows, Ref<UniqueArray<Int32>> target_segments, IThreadBarrier* barrier)
: m_my_rank(my_rank)
, m_sizeof_type(sizeof_type)
, m_ranks(ranks)
, m_windows(windows)
, m_windows_span(windows->smallSpan())
, m_target_segments(target_segments)
, m_target_segments_span(target_segments->smallSpan())
, m_barrier(barrier)
{}

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
segmentView()
{
  return m_windows_span[m_my_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> SharedMemoryDynamicMachineMemoryWindowBaseInternal::
segmentView(Int32 rank)
{
  return m_windows_span[rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> SharedMemoryDynamicMachineMemoryWindowBaseInternal::
segmentConstView() const
{
  return m_windows_span[m_my_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const std::byte> SharedMemoryDynamicMachineMemoryWindowBaseInternal::
segmentConstView(Int32 rank) const
{
  return m_windows_span[rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
add(Span<const std::byte> elem)
{
  m_barrier->wait();
  if (elem.size() % m_sizeof_type != 0) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }
  m_windows_span[m_my_rank].addRange(elem);
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
add()
{
  m_barrier->wait();
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
addToAnotherSegment(Int32 rank, Span<const std::byte> elem)
{
  if (elem.size() % m_sizeof_type != 0) {
    ARCCORE_FATAL("Sizeof elem not valid");
  }

  m_target_segments_span[m_my_rank] = rank;
  m_barrier->wait();

  bool is_found = false;
  for (const Int32 rank_asked : m_target_segments_span) {
    if (rank_asked == rank) {
      if (!is_found) {
        is_found = true;
      }
      else {
        ARCANE_FATAL("Two subdomains ask same rank for addToAnotherSegment()");
      }
    }
  }

  m_windows_span[rank].addRange(elem);
  m_barrier->wait();
  m_target_segments_span[m_my_rank] = -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
addToAnotherSegment()
{
  m_barrier->wait();
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
reserve(Int64 new_capacity)
{
  m_barrier->wait();
  m_windows_span[m_my_rank].reserve(new_capacity);
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
reserve()
{
  m_barrier->wait();
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
resize(Int64 new_size)
{
  m_barrier->wait();
  m_windows_span[m_my_rank].resize(new_size);
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
resize()
{
  m_barrier->wait();
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryDynamicMachineMemoryWindowBaseInternal::
shrink()
{
  m_barrier->wait();
  m_windows_span[m_my_rank].shrink();
  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
