// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineMemoryWindowBaseCreator.h                      (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* HybridMachineMemoryWindowBase. Une instance de cet objet doit être        */
/* partagée par tous les threads d'un processus.                             */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASECREATOR_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASECREATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class MpiParallelMng;

namespace MessagePassing
{
  class IMachineMemoryWindowBase;
  class HybridMachineMemoryWindowBase;

  namespace Mpi
  {
    class MpiMachineMemoryWindowBaseCreator;
  }
} // namespace MessagePassing
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridMachineMemoryWindowBaseCreator
{
 public:

  HybridMachineMemoryWindowBaseCreator(Int32 nb_rank_local_proc, IThreadBarrier* barrier);
  ~HybridMachineMemoryWindowBaseCreator() = default;

 public:

  HybridMachineMemoryWindowBase* createWindow(Int32 my_rank_global, Integer nb_elem_local_proc, Integer sizeof_type, MpiParallelMng* mpi_parallel_mng);

 private:

  void _buildMachineRanksArray(const Mpi::MpiMachineMemoryWindowBaseCreator* mpi_window_creator);

 private:

  Int32 m_nb_rank_local_proc;
  Integer m_nb_elem_total_local_proc;
  IThreadBarrier* m_barrier;
  Ref<IMachineMemoryWindowBase> m_window;
  Ref<IMachineMemoryWindowBase> m_nb_elem;
  Ref<IMachineMemoryWindowBase> m_sum_nb_elem;
  UniqueArray<Int32> m_machine_ranks;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
