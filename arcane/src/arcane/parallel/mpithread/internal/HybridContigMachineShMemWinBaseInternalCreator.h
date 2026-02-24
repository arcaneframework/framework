// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridContigMachineShMemWinBaseInternalCreator.h            (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* HybridContigMachineShMemWinBaseInternal. Une instance de cet objet doit   */
/* être partagée par tous les threads d'un processus.                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDCONTIGMACHINESHMEMWINBASEINTERNALCREATOR_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDCONTIGMACHINESHMEMWINBASEINTERNALCREATOR_H

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
  class IContigMachineShMemWinBaseInternal;
  class HybridContigMachineShMemWinBaseInternal;
  class HybridMachineShMemWinBaseInternal;

  namespace Mpi
  {
    class MpiContigMachineShMemWinBaseInternalCreator;
    class MpiMultiMachineShMemWinBaseInternal;
  } // namespace Mpi
} // namespace MessagePassing
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridContigMachineShMemWinBaseInternalCreator
{
 public:

  HybridContigMachineShMemWinBaseInternalCreator(Int32 nb_rank_local_proc, IThreadBarrier* barrier);
  ~HybridContigMachineShMemWinBaseInternalCreator() = default;

 public:

  HybridContigMachineShMemWinBaseInternal* createWindow(Int32 my_rank_global, Int64 sizeof_segment, Int32 sizeof_type, MpiParallelMng* mpi_parallel_mng);
  HybridMachineShMemWinBaseInternal* createDynamicWindow(Int32 my_rank_global, Int64 sizeof_segment, Int32 sizeof_type, MpiParallelMng* mpi_parallel_mng);

 private:

  void _buildMachineRanksArray(const Mpi::MpiContigMachineShMemWinBaseInternalCreator* mpi_window_creator);

 private:

  Int32 m_nb_rank_local_proc = 0;
  Int64 m_sizeof_segment_local_proc = 0;
  IThreadBarrier* m_barrier = nullptr;
  UniqueArray<Int32> m_machine_ranks;

  Ref<IContigMachineShMemWinBaseInternal> m_window;
  Ref<IContigMachineShMemWinBaseInternal> m_sizeof_sub_segments;
  Ref<IContigMachineShMemWinBaseInternal> m_sum_sizeof_sub_segments;
  //-----------
  Ref<Mpi::MpiMultiMachineShMemWinBaseInternal> m_windows;
  UniqueArray<Int64> m_sizeof_resize_segments;
};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
