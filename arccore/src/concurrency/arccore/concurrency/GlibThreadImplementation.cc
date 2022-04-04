// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 IFPEN-CEA
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
/* GlibThreadImplementation.cc                                 (C) 2000-2021 */
/*                                                                           */
/* Implémentation des threads utilisant la glib.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NotImplementedException.h"
#include "arccore/base/IFunctor.h"

#include "arccore/concurrency/GlibThreadImplementation.h"
#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/concurrency/Mutex.h"
#include "arccore/concurrency/GlibAdapter.h"

#include <new>
#include <glib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static void* _GlibStartFunc(void* f)
{
  IFunctor* ff = reinterpret_cast<IFunctor*>(f);
  ff->executeFunctor();
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GlibThreadBarrier
: public IThreadBarrier
{
 public:

  GlibThreadBarrier()
  : m_wait_mutex(nullptr), m_wait(nullptr), m_nb_thread(0)
  , m_current_reached(0) {}

 public:

  virtual void init(Integer nb_thread)
  {
    m_nb_thread = nb_thread;
    m_current_reached = 0;
    m_wait_mutex = new GlibMutex();
    m_wait = new GlibCond();
  }

  virtual void destroy()
  {
    m_nb_thread = 0;
    m_current_reached = 0;
    delete m_wait_mutex;
    delete m_wait;
    delete this;
  }

  virtual bool wait()
  {
    bool is_last = false;
    m_wait_mutex->lock();
    ++m_current_reached;
    //cout << "ADD BARRIER N=" << m_current_reached << '\n';
    if (m_current_reached==m_nb_thread){
      m_current_reached = 0;
      is_last = true;
      //cout << "BROADCAST BARRIER N=" << m_current_reached << '\n';
      m_wait->broadcast();
    }
    else
      m_wait->wait(m_wait_mutex);
    m_wait_mutex->unlock();
    return is_last;
  }
 private:
  GlibMutex* m_wait_mutex;
  GlibCond* m_wait;
  Integer m_nb_thread;
  Integer m_current_reached;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCCORE_CONCURRENCY_EXPORT IThreadBarrier*
createGlibThreadBarrier()
{
  return new GlibThreadBarrier();
}

namespace
{
inline void _atomicSet(volatile gint* v,int value)
{
  g_atomic_int_set(v,value);
}
//! Retourne true si on acquiere le lock (*v passe de 0 à 1)
inline bool _tryLock(volatile gint* v)
{
  return g_atomic_int_compare_and_exchange(v,0,1)==1;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GlibThreadImplementation::
GlibThreadImplementation()
: m_global_mutex_impl(nullptr)
{
}

GlibThreadImplementation::
~GlibThreadImplementation()
{
  GlobalMutex::destroy();
  if (m_global_mutex_impl)
    destroyMutex(m_global_mutex_impl);
}

void GlibThreadImplementation::
initialize()
{
  m_global_mutex_impl = createMutex();
  GlobalMutex::init(m_global_mutex_impl);
}

ThreadImpl* GlibThreadImplementation::
createThread(IFunctor* f)
{
  return reinterpret_cast<ThreadImpl*>(g_thread_new(nullptr,&_GlibStartFunc,f));
}

void GlibThreadImplementation::
joinThread(ThreadImpl* t)
{
  GThread* tt = reinterpret_cast<GThread*>(t);
  g_thread_join(tt);
}

void GlibThreadImplementation::
destroyThread(ThreadImpl* t)
{
  ARCCORE_UNUSED(t);
}

void GlibThreadImplementation::
createSpinLock(Int64* spin_lock_addr)
{
  volatile gint* v = (gint*)spin_lock_addr;
  _atomicSet(v,0);
}

void GlibThreadImplementation::
lockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr)
{
  ARCCORE_UNUSED(scoped_spin_lock_addr);

  volatile gint* v = (gint*)spin_lock_addr;
  int loop = 0;
  if (!_tryLock(v)){
    do{
      ++loop;
      //TODO: Faire Pause
    } while(!_tryLock(v));
  }
}

void GlibThreadImplementation::
unlockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr)
{
  ARCCORE_UNUSED(scoped_spin_lock_addr);

  volatile gint* v = (gint*)spin_lock_addr;
  _atomicSet(v,0);
}
  
MutexImpl* GlibThreadImplementation::
createMutex()
{
  GlibMutex* m = new GlibMutex();
  return reinterpret_cast<MutexImpl*>(m);
}

void GlibThreadImplementation::
destroyMutex(MutexImpl* mutex)
{
  GlibMutex* m = reinterpret_cast<GlibMutex*>(mutex);
  delete m;
}

void GlibThreadImplementation::
lockMutex(MutexImpl* mutex)
{
  GlibMutex* m = reinterpret_cast<GlibMutex*>(mutex);
  m->lock();
}

void GlibThreadImplementation::
unlockMutex(MutexImpl* mutex)
{
  GlibMutex* m = reinterpret_cast<GlibMutex*>(mutex);
  m->unlock();
}

Int64 GlibThreadImplementation::
currentThread()
{
  Int64 v = reinterpret_cast<Int64>(g_thread_self());
  return v;
}

IThreadBarrier* GlibThreadImplementation::
createBarrier()
{
  return new GlibThreadBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
