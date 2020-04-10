// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiLock.h                                                   (C) 2000-2018 */
/*                                                                           */
/* Verrou pour les appels MPI.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPILOCK_H
#define ARCANE_PARALLEL_MPI_MPILOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/SpinLock.h"
#include "arccore/concurrency/Mutex.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{
namespace Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Verrou pour les appels MPI.
 *
 * Ce verrou sert en multi-threading pour sérialiser les appels
 * MPI en mode MPI_THREAD_SERIALIZED de MPI_Init_thread.
 */
class MpiLock
{
 public:

  // Le spin lock est plus performant mais ne permet pas d'utiliser
  // valgrind.

  //typedef SpinLock LockType;

  typedef Mutex LockType;

 public:
  class Section
  {
   public:
    Section(MpiLock* lock) : mpi_lock(lock)
    {
      if (mpi_lock){
        manual_lock.lock(mpi_lock->m_lock);
      }
    }
    ~Section()
    {
      if (mpi_lock)
        manual_lock.unlock(mpi_lock->m_lock);
    }
   private:
    MpiLock* mpi_lock;
    LockType::ManualLock manual_lock;
  };
  friend class Section;
 public:
  MpiLock() {}
 public:
 private:
  LockType m_lock;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

