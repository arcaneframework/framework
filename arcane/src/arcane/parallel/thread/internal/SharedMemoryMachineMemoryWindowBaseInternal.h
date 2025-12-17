// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBaseInternal.h               (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée.                                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASEINTERNAL_H
#define ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/internal/IMachineMemoryWindowBaseInternal.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_THREAD_EXPORT SharedMemoryMachineMemoryWindowBaseInternal
: public IMachineMemoryWindowBaseInternal
{
 public:

  SharedMemoryMachineMemoryWindowBaseInternal(Int32 my_rank, Int32 nb_rank, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<UniqueArray<std::byte>> window, Ref<UniqueArray<Int64>> sizeof_segments, Ref<UniqueArray<Int64>> sum_sizeof_segments, Int64 sizeof_window, IThreadBarrier* barrier);

  ~SharedMemoryMachineMemoryWindowBaseInternal() override = default;

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

  Int32 m_my_rank = 0;
  Int32 m_nb_rank = 0;
  Int32 m_sizeof_type = 0;
  Int64 m_actual_sizeof_win = 0;
  Int64 m_max_sizeof_win = 0;
  ConstArrayView<Int32> m_ranks;

  Span<std::byte> m_window_span;
  Ref<UniqueArray<std::byte>> m_window;

  Ref<UniqueArray<Int64>> m_sizeof_segments;
  SmallSpan<Int64> m_sizeof_segments_span;

  Ref<UniqueArray<Int64>> m_sum_sizeof_segments;
  SmallSpan<Int64> m_sum_sizeof_segments_span;

  IThreadBarrier* m_barrier = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
