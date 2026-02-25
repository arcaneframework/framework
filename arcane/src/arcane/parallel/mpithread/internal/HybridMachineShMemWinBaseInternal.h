// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMachineShMemWinBaseInternal.h                         (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour l'ensemble des      */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/* Les segments de ces fenêtres ne sont pas contigüs en mémoire et peuvent   */
/* être redimensionnés.                                                      */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINESHMEMWINBASEINTERNAL_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDMACHINESHMEMWINBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arccore/message_passing/internal/IMachineShMemWinBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
namespace Mpi
{
  class MpiMultiMachineShMemWinBaseInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridMachineShMemWinBaseInternal
: public IMachineShMemWinBaseInternal
{
 public:

  HybridMachineShMemWinBaseInternal(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<Mpi::MpiMultiMachineShMemWinBaseInternal> mpi_windows, IThreadBarrier* barrier);

  ~HybridMachineShMemWinBaseInternal() override = default;

 public:

  Int32 sizeofOneElem() const override;
  ConstArrayView<Int32> machineRanks() const override;
  void barrier() const override;

  Span<std::byte> segmentView() override;
  Span<std::byte> segmentView(Int32 rank) override;

  Span<const std::byte> segmentConstView() const override;
  Span<const std::byte> segmentConstView(Int32 rank) const override;

  void add(Span<const std::byte> elem) override;
  void add() override;

  void addToAnotherSegment(Int32 rank, Span<const std::byte> elem) override;
  void addToAnotherSegment() override;

  void reserve(Int64 new_capacity) override;
  void reserve() override;

  void resize(Int64 new_size) override;
  void resize() override;

  void shrink() override;

 private:

  Int32 m_my_rank_local_proc = 0;
  Int32 m_nb_rank_local_proc = 0;
  Int32 m_my_rank_mpi = 0;

  ConstArrayView<Int32> m_machine_ranks;

  Int32 m_sizeof_type = 0;
  Ref<Mpi::MpiMultiMachineShMemWinBaseInternal> m_mpi_windows;

  IThreadBarrier* m_thread_barrier = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
