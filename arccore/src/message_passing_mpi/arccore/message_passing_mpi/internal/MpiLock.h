// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiLock.h                                                   (C) 2000-2025 */
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

namespace Arcane::MessagePassing::Mpi
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

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

