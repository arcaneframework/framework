// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryDynamicMachineMemoryWindowBaseInternal.h        (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée.                                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYDYNAMICMACHINEMEMORYWINDOWBASEINTERNAL_H
#define ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYDYNAMICMACHINEMEMORYWINDOWBASEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/internal/IDynamicMachineMemoryWindowBaseInternal.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_THREAD_EXPORT SharedMemoryDynamicMachineMemoryWindowBaseInternal
: public IDynamicMachineMemoryWindowBaseInternal
{
 public:

  SharedMemoryDynamicMachineMemoryWindowBaseInternal(Int32 my_rank, Int32 nb_rank, ConstArrayView<Int32> ranks, Int32 sizeof_type, UniqueArray<std::byte>* windows, Int32* owner_segments, IThreadBarrier* barrier);

  ~SharedMemoryDynamicMachineMemoryWindowBaseInternal() override;

 public:

  Int32 sizeofOneElem() const override;

  Span<std::byte> segment() override;
  Span<std::byte> segment(Int32 rank) override;

  Int32 segmentOwner() const override;
  Int32 segmentOwner(Int32 rank) const override;

  void add(Span<const std::byte> elem) override;

  void exchangeSegmentWith(Int32 rank) override;
  void exchangeSegmentWith() override;

  void resetExchanges() override;

  ConstArrayView<Int32> machineRanks() const override;

  void syncAdd() override {}

  void barrier() const override;

  void reserve(Int64 new_capacity) override;
  void reserve() override {}

  void resize(Int64 new_size) override;
  void resize() override {}

  void shrink() override;

 private:

  Int32 m_my_rank;
  ConstArrayView<Int32> m_ranks;
  Int32 m_sizeof_type;
  UniqueArray<std::byte>* m_windows;
  SmallSpan<UniqueArray<std::byte>> m_windows_span;
  Int32* m_owner_segments;
  SmallSpan<Int32> m_owner_segments_span;
  IThreadBarrier* m_barrier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
