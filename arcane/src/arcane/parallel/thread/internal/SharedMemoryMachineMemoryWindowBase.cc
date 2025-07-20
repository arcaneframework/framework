// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBase.cc                      (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée.                                        */
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBase.h"

#include "arcane/utils/FatalErrorException.h"

#include "arccore/concurrency/IThreadBarrier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineMemoryWindowBase::
SharedMemoryMachineMemoryWindowBase(Int32 my_rank, Int32 nb_rank, ConstArrayView<Int32> ranks, Integer sizeof_type, std::byte* window, Integer* nb_elem, Integer* sum_nb_elem, Integer nb_elem_total, IThreadBarrier* barrier)
: m_my_rank(my_rank)
, m_nb_rank(nb_rank)
, m_ranks(ranks)
, m_sizeof_type(sizeof_type)
, m_actual_nb_elem_win(nb_elem_total)
, m_max_nb_elem_win(nb_elem_total)
, m_window(window)
, m_nb_elem_segments(nb_elem)
, m_sum_nb_elem_segments(sum_nb_elem)
, m_barrier(barrier)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineMemoryWindowBase::
~SharedMemoryMachineMemoryWindowBase()
{
  if (m_my_rank == 0) {
    delete[] m_window;
    delete[] m_nb_elem_segments;
    delete[] m_sum_nb_elem_segments;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SharedMemoryMachineMemoryWindowBase::
sizeofOneElem() const
{
  return m_sizeof_type;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SharedMemoryMachineMemoryWindowBase::
sizeSegment() const
{
  return m_nb_elem_segments[m_my_rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SharedMemoryMachineMemoryWindowBase::
sizeSegment(Int32 rank) const
{
  return m_nb_elem_segments[rank];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SharedMemoryMachineMemoryWindowBase::
sizeWindow() const
{
  return m_actual_nb_elem_win;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* SharedMemoryMachineMemoryWindowBase::
dataSegment() const
{
  return &m_window[m_sum_nb_elem_segments[m_my_rank] * m_sizeof_type];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* SharedMemoryMachineMemoryWindowBase::
dataSegment(Int32 rank) const
{
  return &m_window[m_sum_nb_elem_segments[rank] * m_sizeof_type];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* SharedMemoryMachineMemoryWindowBase::
dataWindow() const
{
  return m_window;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> SharedMemoryMachineMemoryWindowBase::
sizeAndDataSegment() const
{
  return { sizeSegment(), dataSegment() };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> SharedMemoryMachineMemoryWindowBase::
sizeAndDataSegment(Int32 rank) const
{
  return { sizeSegment(rank), dataSegment(rank) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<Integer, void*> SharedMemoryMachineMemoryWindowBase::
sizeAndDataWindow() const
{
  return { m_actual_nb_elem_win, m_window };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryMachineMemoryWindowBase::
resizeSegment(Integer new_nb_elem)
{
  m_nb_elem_segments[m_my_rank] = new_nb_elem;

  m_barrier->wait();

  if (m_my_rank == 0) {
    Integer sum = 0;
    for (Integer i = 0; i < m_nb_rank; ++i) {
      m_sum_nb_elem_segments[i] = sum;
      sum += m_nb_elem_segments[i];
    }
    if (sum > m_max_nb_elem_win) {
      ARCANE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_nb_elem_win = sum;
  }
  else {
    Integer sum = 0;
    for (Integer i = 0; i < m_nb_rank; ++i) {
      sum += m_nb_elem_segments[i];
    }
    if (sum > m_max_nb_elem_win) {
      ARCANE_FATAL("New size of window (sum of size of all segments) is superior than the old size");
    }
    m_actual_nb_elem_win = sum;
  }

  m_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> SharedMemoryMachineMemoryWindowBase::
machineRanks() const
{
  return m_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
