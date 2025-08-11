// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridDynamicMachineMemoryWindowBaseInternal.h              (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour l'ensemble des      */
/* sous-domaines en mémoire partagée des processus du même noeud.            */
/* Les segments de ces fenêtres ne sont pas contigüs en mémoire et peuvent   */
/* être redimensionnés.                                                      */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDDYNAMICMACHINEMEMORYWINDOWBASEINTERNAL_H
#define ARCANE_PARALLEL_MPITHREAD_INTERNAL_HYBRIDDYNAMICMACHINEMEMORYWINDOWBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arccore/message_passing/internal/IDynamicMachineMemoryWindowBaseInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
namespace Mpi
{
  class MpiDynamicMultiMachineMemoryWindowBaseInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridDynamicMachineMemoryWindowBaseInternal
: public IDynamicMachineMemoryWindowBaseInternal
{
 public:

  HybridDynamicMachineMemoryWindowBaseInternal(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, ConstArrayView<Int32> ranks, Int32 sizeof_type,
                                               Mpi::MpiDynamicMultiMachineMemoryWindowBaseInternal* mpi_windows, IThreadBarrier* barrier);

  ~HybridDynamicMachineMemoryWindowBaseInternal() override;

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

  Int32 m_my_rank_local_proc;
  Int32 m_nb_rank_local_proc;
  Int32 m_my_rank_mpi;

  ConstArrayView<Int32> m_machine_ranks;

  Int32 m_sizeof_type;
  Mpi::MpiDynamicMultiMachineMemoryWindowBaseInternal* m_mpi_windows;

  IThreadBarrier* m_thread_barrier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
