// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryMachineMemoryWindowBaseInternalCreator.h        (C) 2000-2025 */
/*                                                                           */
/* Classe permettant de créer des objets de type                             */
/* SharedMemoryMachineMemoryWindowBaseInternal. Une instance de cet objet    */
/* doit être partagée par tous les threads.                                  */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASEINTERNALCREATOR_H
#define ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYMACHINEMEMORYWINDOWBASEINTERNALCREATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryMachineMemoryWindowBaseInternal;
class SharedMemoryMachineShMemWinBaseInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryMachineMemoryWindowBaseInternalCreator
{
 public:

  SharedMemoryMachineMemoryWindowBaseInternalCreator(Int32 nb_rank, IThreadBarrier* barrier);
  ~SharedMemoryMachineMemoryWindowBaseInternalCreator() = default;

 public:

  SharedMemoryMachineMemoryWindowBaseInternal* createWindow(Int32 my_rank, Int64 sizeof_segment, Int32 sizeof_type);
  SharedMemoryMachineShMemWinBaseInternal* createDynamicWindow(Int32 my_rank, Int64 sizeof_segment, Int32 sizeof_type);

 private:

  Int32 m_nb_rank = 0;
  Int64 m_sizeof_window = 0;
  UniqueArray<Int32> m_ranks;
  IThreadBarrier* m_barrier = nullptr;

  Ref<UniqueArray<std::byte>> m_window;
  Ref<UniqueArray<Int64>> m_sizeof_segments;
  Ref<UniqueArray<Int64>> m_sum_sizeof_segments;
  //-------
  Ref<UniqueArray<UniqueArray<std::byte>>> m_windows;
  Ref<UniqueArray<Int32>> m_target_segments;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
