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

  SharedMemoryMachineMemoryWindowBaseInternal(Int32 my_rank, Int32 nb_rank, ConstArrayView<Int32> ranks, Int32 sizeof_type, std::byte* window, Int64* sizeof_segments, Int64* sum_sizeof_segments, Int64 sizeof_window, IThreadBarrier* barrier);

  ~SharedMemoryMachineMemoryWindowBaseInternal() override;

 public:

  Int32 sizeofOneElem() const override;

  Span<std::byte> segment() const override;
  Span<std::byte> segment(Int32 rank) const override;
  Span<std::byte> window() const override;

  void resizeSegment(Int64 new_sizeof_segment) override;

  ConstArrayView<Int32> machineRanks() const override;

  void barrier() const override;

 private:

  Int32 m_my_rank;
  Int32 m_nb_rank;
  ConstArrayView<Int32> m_ranks;
  Int32 m_sizeof_type;
  Int64 m_actual_sizeof_win;
  Int64 m_max_sizeof_win;
  Span<std::byte> m_window_span;
  Span<Int64> m_sizeof_segments_span;
  Span<Int64> m_sum_sizeof_segments_span;
  std::byte* m_window;
  Int64* m_sizeof_segments;
  Int64* m_sum_sizeof_segments;
  IThreadBarrier* m_barrier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
