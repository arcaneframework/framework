// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBaseCreator.cc               (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* SharedMemoryMachineMemoryWindowBase. Une instance de cet objet doit être  */
/* partagée par tous les threads.                                            */
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBaseCreator.h"

#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBase.h"
#include "arccore/concurrency/IThreadBarrier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryMachineMemoryWindowBaseCreator::
SharedMemoryMachineMemoryWindowBaseCreator(Int32 nb_rank, IThreadBarrier* barrier)
: m_nb_rank(nb_rank)
, m_nb_elem_total(0)
, m_barrier(barrier)
, m_window(nullptr)
, m_nb_elem(nullptr)
, m_sum_nb_elem(nullptr)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMachineMemoryWindowBase* SharedMemoryMachineMemoryWindowBaseCreator::
createWindow(Int32 my_rank, Integer nb_elem_local_section, Integer sizeof_type)
{
  if (my_rank == 0) {
    m_nb_elem = new Integer[m_nb_rank];
    m_sum_nb_elem = new Integer[m_nb_rank];
  }
  m_barrier->wait();

  m_nb_elem[my_rank] = nb_elem_local_section;
  m_barrier->wait();

  if (my_rank == 0) {
    m_nb_elem_total = 0;
    for (Integer i = 0; i < m_nb_rank; ++i) {
      m_sum_nb_elem[i] = m_nb_elem_total;
      m_nb_elem_total += m_nb_elem[i];
    }
    m_window = new std::byte[m_nb_elem_total * sizeof_type];
  }
  m_barrier->wait();

  auto* window_obj = new SharedMemoryMachineMemoryWindowBase(my_rank, m_nb_rank, sizeof_type, m_window, m_nb_elem, m_sum_nb_elem, m_nb_elem_total, m_barrier);
  m_barrier->wait();

  // Ces tableaux doivent être delete par SharedMemoryMachineMemoryWindowBase (rang 0 uniquement).
  m_nb_elem = nullptr;
  m_sum_nb_elem = nullptr;
  m_window = nullptr;
  m_nb_elem_total = 0;

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
