﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryMachineMemoryWindowBaseInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryMachineMemoryWindowBaseInternalCreator
{
 public:

  SharedMemoryMachineMemoryWindowBaseInternalCreator(Int32 nb_rank, IThreadBarrier* barrier);
  ~SharedMemoryMachineMemoryWindowBaseInternalCreator() = default;

 public:

  SharedMemoryMachineMemoryWindowBaseInternal* createWindow(Int32 my_rank, Int64 sizeof_segment, Int32 sizeof_type);

 private:

  Int32 m_nb_rank;
  Int64 m_sizeof_window;
  IThreadBarrier* m_barrier;
  std::byte* m_window;
  Int64* m_sizeof_segments;
  Int64* m_sum_sizeof_segments;
  UniqueArray<Int32> m_ranks;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
