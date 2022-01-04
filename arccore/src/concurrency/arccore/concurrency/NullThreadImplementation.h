// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NullThreadImplementation.h                                  (C) 2000-2019 */
/*                                                                           */
/* Gestionnaire de thread en mode mono-thread.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_NULLTHREADIMPLEMENTATION_H
#define ARCCORE_CONCURRENCY_NULLTHREADIMPLEMENTATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/concurrency/IThreadImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'une barrière en mono-thread.
 */
class ARCCORE_CONCURRENCY_EXPORT NullThreadBarrier
: public IThreadBarrier
{
  virtual void init(Integer nb_thread) { ARCCORE_UNUSED(nb_thread); }
  virtual void destroy() { delete this; }
  virtual bool wait() { return true; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation des threads en mode mono-thread.
 */
class ARCCORE_CONCURRENCY_EXPORT NullThreadImplementation
: public IThreadImplementation
{
 public:
  virtual void initialize(){}
 public:
  virtual void addReference() {}
  virtual void removeReference() {}
  virtual ThreadImpl* createThread(IFunctor*) { return nullptr; }
  virtual void joinThread(ThreadImpl*) {}
  virtual void destroyThread(ThreadImpl*) {}

  virtual void createSpinLock(Int64* spin_lock_addr)
  {
    ARCCORE_UNUSED(spin_lock_addr);
  }
  virtual void lockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr)
  {
    ARCCORE_UNUSED(spin_lock_addr);
    ARCCORE_UNUSED(scoped_spin_lock_addr);
  }
  virtual void unlockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr)
  {
    ARCCORE_UNUSED(spin_lock_addr);
    ARCCORE_UNUSED(scoped_spin_lock_addr);
  }

  virtual MutexImpl* createMutex() { return nullptr; }
  virtual void destroyMutex(MutexImpl*) {}
  virtual void lockMutex(MutexImpl*) {}
  virtual void unlockMutex(MutexImpl*) {}

  virtual Int64 currentThread() { return 0; }

  virtual IThreadBarrier* createBarrier() { return new NullThreadBarrier(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
