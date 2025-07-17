// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBaseCreator.h                (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* SharedMemoryMachineMemoryWindowBase. Une instance de cet objet doit être  */
/* partagée par tous les threads.                                            */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASECREATOR_H
#define ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASECREATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMachineMemoryWindowBase;
class SharedMemoryMachineMemoryWindowBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryMachineMemoryWindowBaseCreator
{
 public:

  SharedMemoryMachineMemoryWindowBaseCreator(Int32 nb_rank, IThreadBarrier* barrier);
  ~SharedMemoryMachineMemoryWindowBaseCreator() = default;

 public:

  IMachineMemoryWindowBase* createWindow(Int32 my_rank, Integer nb_elem_local_section, Integer sizeof_type);

 private:

  Int32 m_nb_rank;
  Integer m_nb_elem_total;
  IThreadBarrier* m_barrier;
  std::byte* m_window;
  Integer* m_nb_elem;
  Integer* m_sum_nb_elem;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
