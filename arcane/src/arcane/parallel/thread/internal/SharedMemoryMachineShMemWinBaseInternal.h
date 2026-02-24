// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineShMemWinBaseInternal.h        (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des fenêtres mémoires pour l'ensemble des      */
/* sous-domaines en mémoire partagée.                                        */
/* Les segments de ces fenêtres ne sont pas contigüs en mémoire et peuvent   */
/* être redimensionnés.                                                      */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINESHMEMWINBASEINTERNAL_H
#define ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINESHMEMWINBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/internal/IMachineShMemWinBaseInternal.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_THREAD_EXPORT SharedMemoryMachineShMemWinBaseInternal
: public IMachineShMemWinBaseInternal
{
 public:

  SharedMemoryMachineShMemWinBaseInternal(Int32 my_rank, ConstArrayView<Int32> ranks, Int32 sizeof_type, Ref<UniqueArray<UniqueArray<std::byte>>> windows, Ref<UniqueArray<Int32>> target_segments, IThreadBarrier* barrier);

  ~SharedMemoryMachineShMemWinBaseInternal() override = default;

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

  Int32 m_my_rank = 0;
  Int32 m_sizeof_type = 0;
  ConstArrayView<Int32> m_ranks;

  Ref<UniqueArray<UniqueArray<std::byte>>> m_windows;
  SmallSpan<UniqueArray<std::byte>> m_windows_span;

  Ref<UniqueArray<Int32>> m_target_segments;
  SmallSpan<Int32> m_target_segments_span;

  IThreadBarrier* m_barrier = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
