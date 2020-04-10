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
