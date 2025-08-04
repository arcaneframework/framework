// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineMemoryWindowBaseInternal.h                     (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASEINTERNAL_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINEMEMORYWINDOWBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arccore/message_passing/internal/IMachineMemoryWindowBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridMachineMemoryWindowBaseInternal
: public IMachineMemoryWindowBaseInternal
{
 public:

  HybridMachineMemoryWindowBaseInternal(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<IMachineMemoryWindowBaseInternal> nb_elem, Ref<IMachineMemoryWindowBaseInternal> sum_nb_elem, Ref<IMachineMemoryWindowBaseInternal> mpi_window, IThreadBarrier* barrier);

  ~HybridMachineMemoryWindowBaseInternal() override = default;

 public:

  Int32 sizeofOneElem() const override;

  Span<std::byte> segmentView() override;
  Span<std::byte> segmentView(Int32 rank) override;
  Span<std::byte> windowView() override;

  Span<const std::byte> segmentConstView() const override;
  Span<const std::byte> segmentConstView(Int32 rank) const override;
  Span<const std::byte> windowConstView() const override;

  void resizeSegment(Int64 new_sizeof_segment) override;

  ConstArrayView<Int32> machineRanks() const override;

  void barrier() const override;

 private:

  Int32 m_my_rank_local_proc;
  Int32 m_nb_rank_local_proc;
  Int32 m_my_rank_mpi;
  ConstArrayView<Int32> m_machine_ranks;
  Int32 m_sizeof_type;
  Ref<IMachineMemoryWindowBaseInternal> m_mpi_window;
  Ref<IMachineMemoryWindowBaseInternal> m_sizeof_sub_segments_global;
  Ref<IMachineMemoryWindowBaseInternal> m_sum_sizeof_sub_segments_global;
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
