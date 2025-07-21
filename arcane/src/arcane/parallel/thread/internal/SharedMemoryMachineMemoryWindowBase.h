// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBase.h                       (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer une fenêtre mémoire pour l'ensemble des        */
/* sous-domaines en mémoire partagée.                                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASE_H
#define ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMachineMemoryWindowBase.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_THREAD_EXPORT SharedMemoryMachineMemoryWindowBase
: public IMachineMemoryWindowBase
{
 public:

  SharedMemoryMachineMemoryWindowBase(Int32 my_rank, Int32 nb_rank, ConstArrayView<Int32> ranks, Integer sizeof_type, std::byte* window, Integer* nb_elem, Integer* sum_nb_elem, Integer nb_elem_total, IThreadBarrier* barrier);

  ~SharedMemoryMachineMemoryWindowBase() override;

 public:

  Integer sizeofOneElem() const override;

  Integer sizeSegment() const override;
  Integer sizeSegment(Int32 rank) const override;

  Integer sizeWindow() const override;

  void* dataSegment() const override;
  void* dataSegment(Int32 rank) const override;

  void* dataWindow() const override;

  std::pair<Integer, void*> sizeAndDataSegment() const override;
  std::pair<Integer, void*> sizeAndDataSegment(Int32 rank) const override;

  std::pair<Integer, void*> sizeAndDataWindow() const override;

  void resizeSegment(Integer new_size) override;

  ConstArrayView<Int32> machineRanks() const override;

  void barrier() const override;

 private:

  Int32 m_my_rank;
  Int32 m_nb_rank;
  ConstArrayView<Int32> m_ranks;
  Integer m_sizeof_type;
  Integer m_actual_nb_elem_win;
  Integer m_max_nb_elem_win;
  std::byte* m_window;
  Integer* m_nb_elem_segments;
  Integer* m_sum_nb_elem_segments;
  IThreadBarrier* m_barrier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
