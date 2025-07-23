// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBaseInternalCreator.cc       (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* SharedMemoryMachineMemoryWindowBaseInternal. Une instance de cet objet    */
/* doit être partagée par tous les threads.                                  */
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBaseInternalCreator.h"

#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBaseInternal.h"
#include "arccore/concurrency/IThreadBarrier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineMemoryWindowBaseInternalCreator::
SharedMemoryMachineMemoryWindowBaseInternalCreator(Int32 nb_rank, IThreadBarrier* barrier)
: m_nb_rank(nb_rank)
, m_sizeof_window(0)
, m_barrier(barrier)
, m_window(nullptr)
, m_sizeof_segments(nullptr)
, m_sum_sizeof_segments(nullptr)
{
  m_ranks.resize(m_nb_rank);
  for (Int32 i = 0; i < m_nb_rank; ++i) {
    m_ranks[i] = i;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineMemoryWindowBaseInternal* SharedMemoryMachineMemoryWindowBaseInternalCreator::
createWindow(Int32 my_rank, Int64 sizeof_segment, Int32 sizeof_type)
{
  if (my_rank == 0) {
    m_sizeof_segments = new Int64[m_nb_rank];
    m_sum_sizeof_segments = new Int64[m_nb_rank];
  }
  m_barrier->wait();

  m_sizeof_segments[my_rank] = sizeof_segment;
  m_barrier->wait();

  if (my_rank == 0) {
    m_sizeof_window = 0;
    for (Int32 i = 0; i < m_nb_rank; ++i) {
      m_sum_sizeof_segments[i] = m_sizeof_window;
      m_sizeof_window += m_sizeof_segments[i];
    }
    m_window = new std::byte[m_sizeof_window];
  }
  m_barrier->wait();

  auto* window_obj = new SharedMemoryMachineMemoryWindowBaseInternal(my_rank, m_nb_rank, m_ranks, sizeof_type, m_window, m_sizeof_segments, m_sum_sizeof_segments, m_sizeof_window, m_barrier);
  m_barrier->wait();

  // Ces tableaux doivent être delete par SharedMemoryMachineMemoryWindowBaseInternal (rang 0 uniquement).
  m_sizeof_segments = nullptr;
  m_sum_sizeof_segments = nullptr;
  m_window = nullptr;
  m_sizeof_window = 0;

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
