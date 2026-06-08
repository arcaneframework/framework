// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiLock.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Lock for MPI calls.                                                       */
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
 * \brief Lock for MPI calls.
 *
 * This lock is used in multi-threading to serialize MPI calls in
 * MPI_THREAD_SERIALIZED mode of MPI_Init_thread.
 */
class MpiLock
{
 public:

  // The spin lock is more performant but does not allow using valgrind.

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
