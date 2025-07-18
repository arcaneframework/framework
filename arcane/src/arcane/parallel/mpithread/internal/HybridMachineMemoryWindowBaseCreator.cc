// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineMemoryWindowBaseCreator.cc                     (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* HybridMachineMemoryWindowBase. Une instance de cet objet doit être        */
/* partagée par tous les threads d'un processus.                             */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBaseCreator.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"
#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBase.h"
#include "arcane/parallel/mpithread/HybridMessageQueue.h"

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMachineMemoryWindowBaseCreator::
HybridMachineMemoryWindowBaseCreator(Int32 nb_rank_local_proc, IThreadBarrier* barrier)
: m_nb_rank_local_proc(nb_rank_local_proc)
, m_nb_elem_total_local_proc(0)
, m_barrier(barrier)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMachineMemoryWindowBase* HybridMachineMemoryWindowBaseCreator::
createWindow(Int32 my_rank_global, Integer nb_elem_local_proc, Integer sizeof_type, MpiParallelMng* mpi_parallel_mng)
{
  // On est dans un contexte où chaque processus doit avoir plusieurs segments, un par thread.
  // Pour que chaque processus puisse avoir accès à toutes les positions des segments de tous les
  // threads de tous les processus, chaque processus doit partager les positions de ces segments avec
  // les autres processus. Pour faire ça, on utilise des fenêtres mémoire MPI.

  FullRankInfo my_fri = FullRankInfo::compute(MP::MessageRank(my_rank_global), m_nb_rank_local_proc);
  Int32 my_rank_local_proc = my_fri.localRankValue();
  Int32 my_rank_mpi = my_fri.mpiRankValue();

  if (my_rank_local_proc == 0) {
    // Le nombre d'éléments de chaque segment. Cette fenêtre fera une taille de nb_thread * nb_proc_sur_le_même_noeud.
    m_nb_elem = mpi_parallel_mng->adapter()->createMachineMemoryWindowBase(m_nb_rank_local_proc, sizeof(Int32));
    m_sum_nb_elem = mpi_parallel_mng->adapter()->createMachineMemoryWindowBase(m_nb_rank_local_proc, sizeof(Int32));
  }
  m_barrier->wait();

  // nb_elem est le segment de notre processus (qui contient les segments de tous nos threads).
  Int32* nb_elem = static_cast<Int32*>(m_nb_elem->data());

  nb_elem[my_rank_local_proc] = nb_elem_local_proc;
  m_barrier->wait();

  if (my_rank_local_proc == 0) {
    m_nb_elem_total_local_proc = 0;

    Int32* sum_nb_elem = static_cast<Int32*>(m_sum_nb_elem->data());

    for (Integer i = 0; i < m_nb_rank_local_proc; ++i) {
      sum_nb_elem[i] = m_nb_elem_total_local_proc;
      m_nb_elem_total_local_proc += nb_elem[i];
    }
  }
  m_barrier->wait();


  if (my_rank_local_proc == 0) {
    m_window = mpi_parallel_mng->adapter()->createMachineMemoryWindowBase(m_nb_elem_total_local_proc, sizeof_type);
  }
  m_barrier->wait();

  auto* window_obj = new HybridMachineMemoryWindowBase(my_rank_mpi, my_rank_local_proc, m_nb_rank_local_proc, sizeof_type, m_nb_elem, m_sum_nb_elem, m_window, m_barrier);
  m_barrier->wait();

  // Ces tableaux doivent être delete par HybridMachineMemoryWindowBase (rang 0 uniquement).
  m_nb_elem.reset();
  m_sum_nb_elem.reset();
  m_nb_elem_total_local_proc = 0;
  m_window.reset();

  return window_obj;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
