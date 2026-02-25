// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridContigMachineShMemWinBaseInternal.h                   (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDCONTIGMACHINESHMEMWINBASEINTERNAL_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDCONTIGMACHINESHMEMWINBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arccore/message_passing/internal/IContigMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridContigMachineShMemWinBaseInternal
: public IContigMachineShMemWinBaseInternal
{
 public:

  HybridContigMachineShMemWinBaseInternal(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<IContigMachineShMemWinBaseInternal> nb_elem, Ref<IContigMachineShMemWinBaseInternal> sum_nb_elem, Ref<IContigMachineShMemWinBaseInternal> mpi_window, IThreadBarrier* barrier);

  ~HybridContigMachineShMemWinBaseInternal() override = default;

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

  Int32 m_my_rank_local_proc = 0;
  Int32 m_nb_rank_local_proc = 0;
  Int32 m_my_rank_mpi = 0;
  ConstArrayView<Int32> m_machine_ranks;
  Int32 m_sizeof_type = 0;
  Ref<IContigMachineShMemWinBaseInternal> m_mpi_window;
  Ref<IContigMachineShMemWinBaseInternal> m_sizeof_sub_segments_global;
  Ref<IContigMachineShMemWinBaseInternal> m_sum_sizeof_sub_segments_global;
  Span<Int64> m_sizeof_sub_segments_local_proc;
  Span<Int64> m_sum_sizeof_sub_segments_local_proc;
  IThreadBarrier* m_thread_barrier = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
