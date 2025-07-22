// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineMemoryWindowBase.h                             (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASE_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arccore/message_passing/internal/IMachineMemoryWindowBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridMachineMemoryWindowBase
: public IMachineMemoryWindowBase
{
 public:

  HybridMachineMemoryWindowBase(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<IMachineMemoryWindowBase> nb_elem, Ref<IMachineMemoryWindowBase> sum_nb_elem, Ref<IMachineMemoryWindowBase> mpi_window, IThreadBarrier* barrier);

  ~HybridMachineMemoryWindowBase() override = default;

 public:

  Int32 sizeofOneElem() const override;

  Span<std::byte> segment() const override;
  Span<std::byte> segment(Int32 rank) const override;
  Span<std::byte> window() const override;

  void resizeSegment(Int64 new_sizeof_segment) override;

  ConstArrayView<Int32> machineRanks() const override;

  void barrier() const override;

 private:

  Int32 m_my_rank_local_proc;
  Int32 m_nb_rank_local_proc;
  Int32 m_my_rank_mpi;
  ConstArrayView<Int32> m_machine_ranks;
  Int32 m_sizeof_type;
  Ref<IMachineMemoryWindowBase> m_mpi_window;
  Ref<IMachineMemoryWindowBase> m_sizeof_sub_segments_global;
  Ref<IMachineMemoryWindowBase> m_sum_sizeof_sub_segments_global;
  Span<Int64> m_sizeof_sub_segments_local_proc;
  Span<Int64> m_sum_sizeof_sub_segments_local_proc;
  IThreadBarrier* m_thread_barrier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
