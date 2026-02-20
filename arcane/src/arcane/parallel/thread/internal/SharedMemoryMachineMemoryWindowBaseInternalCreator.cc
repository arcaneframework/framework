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
#include "arcane/parallel/thread/internal/SharedMemoryMachineShMemWinBaseInternal.h"
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
, m_barrier(barrier)
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
    m_sizeof_segments = makeRef(new UniqueArray<Int64>(m_nb_rank));
    m_sum_sizeof_segments = makeRef(new UniqueArray<Int64>(m_nb_rank));
  }
  m_barrier->wait();

  (*m_sizeof_segments.get())[my_rank] = sizeof_segment;
  m_barrier->wait();

  if (my_rank == 0) {
    m_sizeof_window = 0;
    for (Int32 i = 0; i < m_nb_rank; ++i) {
      (*m_sum_sizeof_segments.get())[i] = m_sizeof_window;
      m_sizeof_window += (*m_sizeof_segments.get())[i];
    }
    m_window = makeRef(new UniqueArray<std::byte>(m_sizeof_window));
  }
  m_barrier->wait();

  auto* window_obj = new SharedMemoryMachineMemoryWindowBaseInternal(my_rank, m_nb_rank, m_ranks, sizeof_type, m_window, m_sizeof_segments, m_sum_sizeof_segments, m_sizeof_window, m_barrier);
  m_barrier->wait();

  // Ces tableaux doivent être delete par SharedMemoryMachineMemoryWindowBaseInternal.
  if (my_rank == 0) {
    m_sizeof_segments.reset();
    m_sum_sizeof_segments.reset();
    m_window.reset();
    m_sizeof_window = 0;
  }

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineShMemWinBaseInternal* SharedMemoryMachineMemoryWindowBaseInternalCreator::
createDynamicWindow(Int32 my_rank, Int64 sizeof_segment, Int32 sizeof_type)
{
  if (my_rank == 0) {
    m_windows = makeRef(new UniqueArray<UniqueArray<std::byte>>(m_nb_rank));
    m_target_segments = makeRef(new UniqueArray<Int32>(m_nb_rank));
  }
  m_barrier->wait();

  (*m_windows.get())[my_rank].resize(sizeof_segment);
  (*m_target_segments.get())[my_rank] = -1;

  auto* window_obj = new SharedMemoryMachineShMemWinBaseInternal(my_rank, m_ranks, sizeof_type, m_windows, m_target_segments, m_barrier);
  m_barrier->wait();

  // Ces tableaux doivent être delete par SharedMemoryMachineShMemWinBaseInternal.
  if (my_rank == 0) {
    m_windows.reset();
    m_target_segments.reset();
  }

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
