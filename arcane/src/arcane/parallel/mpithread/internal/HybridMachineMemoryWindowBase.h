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

#include "arccore/message_passing/IMachineMemoryWindowBase.h"

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

  HybridMachineMemoryWindowBase(Int32 my_rank_mpi, Int32 my_rank_local_proc, Int32 nb_rank_local_proc, Integer sizeof_type, Ref<IMachineMemoryWindowBase> nb_elem, Ref<IMachineMemoryWindowBase> sum_nb_elem, Ref<IMachineMemoryWindowBase> mpi_window, IThreadBarrier* barrier);

  ~HybridMachineMemoryWindowBase() override = default;

 public:

  Integer sizeofOneElem() const override;

  Integer sizeSegment() const override;
  Integer sizeSegment(Int32 rank) const override;

  void* data() const override;
  void* data(Int32 rank) const override;

  std::pair<Integer, void*> sizeAndDataSegment() const override;
  std::pair<Integer, void*> sizeAndDataSegment(Int32 rank) const override;

  void resizeSegment(Integer new_nb_elem) override;

 private:

  Int32 m_my_rank_local_proc;
  Int32 m_nb_rank_local_proc;
  Int32 m_my_rank_mpi;
  Integer m_sizeof_type;
  Ref<IMachineMemoryWindowBase> m_mpi_window;
  Ref<IMachineMemoryWindowBase> m_nb_elem_global;
  Ref<IMachineMemoryWindowBase> m_sum_nb_elem_global;
  Integer* m_nb_elem_local_proc;
  Integer* m_sum_nb_elem_local_proc;
  IThreadBarrier* m_thread_barrier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
