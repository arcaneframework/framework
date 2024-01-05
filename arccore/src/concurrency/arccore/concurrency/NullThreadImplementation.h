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

  void initialize() override {}
  void addReference() override {}
  void removeReference() override {}
  ThreadImpl* createThread(IFunctor*) override { return nullptr; }
  void joinThread(ThreadImpl*) override {}
  void destroyThread(ThreadImpl*) override {}

  void createSpinLock(Int64* spin_lock_addr) override
  {
    ARCCORE_UNUSED(spin_lock_addr);
  }
  void lockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr) override
  {
    ARCCORE_UNUSED(spin_lock_addr);
    ARCCORE_UNUSED(scoped_spin_lock_addr);
  }
  void unlockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr) override
  {
    ARCCORE_UNUSED(spin_lock_addr);
    ARCCORE_UNUSED(scoped_spin_lock_addr);
  }

  MutexImpl* createMutex() override { return nullptr; }
  void destroyMutex(MutexImpl*) override {}
  void lockMutex(MutexImpl*) override {}
  void unlockMutex(MutexImpl*) override {}

  Int64 currentThread() override { return 0; }

  IThreadBarrier* createBarrier() override { return new NullThreadBarrier(); }

  bool isMultiThread() const override { return false; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
