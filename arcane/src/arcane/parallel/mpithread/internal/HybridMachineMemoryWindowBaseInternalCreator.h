// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineMemoryWindowBaseInternalCreator.h              (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* HybridMachineMemoryWindowBaseInternal. Une instance de cet objet doit     */
/* être partagée par tous les threads d'un processus.                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASEINTERNALCREATOR_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASEINTERNALCREATOR_H

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
  class IMachineMemoryWindowBaseInternal;
  class HybridMachineMemoryWindowBaseInternal;
  class HybridDynamicMachineMemoryWindowBaseInternal;

  namespace Mpi
  {
    class MpiMachineMemoryWindowBaseInternalCreator;
    class MpiDynamicMultiMachineMemoryWindowBaseInternal;
  } // namespace Mpi
} // namespace MessagePassing
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridMachineMemoryWindowBaseInternalCreator
{
 public:

  HybridMachineMemoryWindowBaseInternalCreator(Int32 nb_rank_local_proc, IThreadBarrier* barrier);
  ~HybridMachineMemoryWindowBaseInternalCreator() = default;

 public:

  HybridMachineMemoryWindowBaseInternal* createWindow(Int32 my_rank_global, Int64 sizeof_segment, Int32 sizeof_type, MpiParallelMng* mpi_parallel_mng);
  HybridDynamicMachineMemoryWindowBaseInternal* createDynamicWindow(Int32 my_rank_global, Int64 sizeof_segment, Int32 sizeof_type, MpiParallelMng* mpi_parallel_mng);

 private:

  void _buildMachineRanksArray(const Mpi::MpiMachineMemoryWindowBaseInternalCreator* mpi_window_creator);

 private:

  Int32 m_nb_rank_local_proc;
  Int64 m_sizeof_segment_local_proc;
  IThreadBarrier* m_barrier;
  Ref<IMachineMemoryWindowBaseInternal> m_window;
  Ref<IMachineMemoryWindowBaseInternal> m_sizeof_sub_segments;
  Ref<IMachineMemoryWindowBaseInternal> m_sum_sizeof_sub_segments;
  UniqueArray<Int32> m_machine_ranks;
  //-----------
  Mpi::MpiDynamicMultiMachineMemoryWindowBaseInternal* m_windows;
  std::unique_ptr<Int64[]> m_sizeof_resize_segments;
};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
