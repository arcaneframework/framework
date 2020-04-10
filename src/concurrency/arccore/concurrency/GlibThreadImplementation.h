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
/* GlibThreadImplementation.h                                  (C) 2000-2019 */
/*                                                                           */
/* Implémentation de ITreadImplementation avec la 'Glib'.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_GLIBTHREADIMPLEMENTATION_H
#define ARCCORE_CONCURRENCY_GLIBTHREADIMPLEMENTATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadImplementation.h"
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de ITreadImplementation avec la 'Glib'.
 */
class ARCCORE_CONCURRENCY_EXPORT GlibThreadImplementation
: public IThreadImplementation
{
 public:
  GlibThreadImplementation();
  ~GlibThreadImplementation() override;
 public:
  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
 public:
  void initialize() override;
 public:
  ThreadImpl* createThread(IFunctor* f) override;
  void joinThread(ThreadImpl* t) override;
  void destroyThread(ThreadImpl* t) override;

  void createSpinLock(Int64* spin_lock_addr) override;
  void lockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr) override;
  void unlockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr) override;

  MutexImpl* createMutex() override;
  void destroyMutex(MutexImpl*) override;
  void lockMutex(MutexImpl* mutex) override;
  void unlockMutex(MutexImpl* mutex) override;

  Int64 currentThread() override;

  IThreadBarrier* createBarrier() override;
 private:
  MutexImpl* m_global_mutex_impl;
  std::atomic<int> m_nb_ref = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
