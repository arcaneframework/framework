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
/* Mutex.h                                                     (C) 2000-2019 */
/*                                                                           */
/* Mutex pour le multi-threading.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_MUTEX_H
#define ARCCORE_CONCURRENCY_MUTEX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"
#include "arccore/concurrency/IThreadImplementation.h"
#include "arccore/base/ReferenceCounter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MutexImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mutex.
 */
class ARCCORE_CONCURRENCY_EXPORT Mutex
{
 public:
  class ScopedLock
  {
   public:
    ScopedLock(Mutex& m)
    : m_p(&m)
    {
      m_p->lock();
    }
    ScopedLock(Mutex* m)
    : m_p(m)
    {
      m_p->lock();
    }
    ~ScopedLock()
    {
      m_p->unlock();
    }
   private:
    Mutex* m_p;
  };

  class ManualLock
  {
   public:
    void lock(Mutex& m)
    {
      m.lock();
    }
    void unlock(Mutex& m)
    {
      m.unlock();
    }
   private:
  };

  friend class ScopedLock;
  friend class ManualLock;

 public:
  Mutex();
  ~Mutex();
 private:
  void lock();
  void unlock();
 private:
  MutexImpl* m_p;
  //! Implémentation utilisée pour ce mutex.
  ReferenceCounter<IThreadImplementation> m_thread_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mutex global.
 */
class ARCCORE_CONCURRENCY_EXPORT GlobalMutex
{
 public:
  class ScopedLock
  {
   public:
    ScopedLock()
    {
      GlobalMutex::lock();
    }
    ~ScopedLock()
    {
      GlobalMutex::unlock();
    }
  };
  friend class ScopedLock;
 public:
  GlobalMutex(){}
  ~GlobalMutex() {}
 public:
  //! Initialise le mutex global. Interne a Arccore. Doit être alloué par new
  static void init(MutexImpl* p);
  static void lock();
  static void unlock();
  static void destroy();
 private:
  static MutexImpl* m_p;
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
