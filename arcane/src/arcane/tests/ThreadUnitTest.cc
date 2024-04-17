// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadUnitTest.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Service de test des threads.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/SpinLock.h"
#include "arcane/utils/Mutex.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/ThreadUnitTest_axl.h"

#include "arcane/utils/Functor.h"
#include "arcane/utils/IThreadImplementation.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Atomic.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

class MyThread
{
 public:
  MyThread(IFunctor* f)
  {
    create(f);
  }
  MyThread() : m_functor(nullptr), m_thread(nullptr) {}
  ~MyThread()
  {
    if (m_thread)
      platform::getThreadImplementationService()->destroyThread(m_thread);
  }
  void create(IFunctor* f)
  {
    m_functor = f;
    m_thread = platform::getThreadImplementationService()->createThread(f);
  }
  void join()
  {
    platform::getThreadImplementationService()->joinThread(m_thread);
  }

 public:
  IFunctor* m_functor;
  ThreadImpl* m_thread;
};

namespace ThreadTest
{
class Test1
: public TraceAccessor
{
 public:

  explicit Test1(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

  void exec()
  {
    m_atomic_add_count = 0;
    m_atomic_sub_count = 3000;
    AtomicInt32::setValue(&m_add_count,0);
    AtomicInt32::setValue(&m_sub_count,3000);
    for( Integer i=0; i<100; ++i ){
      FunctorT<Test1> f1(this,&Test1::_F1);
      constexpr Integer nb = 10;
      std::vector<MyThread> threads(nb);
      for( Integer j=0; j<nb; ++j )
        threads[j].create(&f1);
      for( Integer j=0; j<nb; ++j )
        threads[j].join();
    }
    info() << "ATOMIC_ADD_COUNT=" << m_atomic_add_count.value();
    info() << "ATOMIC_SUB_COUNT=" << m_atomic_sub_count.value();
    info() << "ADD_COUNT=" << m_add_count;
    info() << "SUB_COUNT=" << m_sub_count;

    if (m_count.load()!=1000)
      ARCANE_FATAL("Bad value for atomic count: v={0} expected=1000",m_count.load());
    if (m_count2!=10000)
      ARCANE_FATAL("Bad value for count2: v={0} expected=10000",m_count2);

    if (m_atomic_add_count.value()!=3000)
      ARCANE_FATAL("Bad value for atomic m_atomic_add_count: v={0} expected=3000",m_atomic_add_count.value());
    if (m_atomic_sub_count.value()!=0)
      ARCANE_FATAL("Bad value for atomic m_atomic_sub_count: v={0} expected=0",m_atomic_sub_count.value());

    if (m_add_count!=3000)
      ARCANE_FATAL("Bad value for atomic m_add_count: v={0} expected=3000",m_add_count);
    if (m_sub_count!=0)
      ARCANE_FATAL("Bad value for atomic m_sub_count: v={0} expected=0",m_sub_count);
  }
  void _F1()
  {
    ++m_count;
    {
      SpinLock::ScopedLock sl(m_lock);
      m_count2 += 10;
    }
    for( Int32 i=0;i<3; ++i ){
      ++m_atomic_add_count;
      AtomicInt32::increment(&m_add_count);
    }
    for( Int32 i=0;i<3; ++i ){
      --m_atomic_sub_count;
      AtomicInt32::decrement(&m_sub_count);
    }
    info() << "MY_THREAD=" << platform::getThreadImplementationService()->currentThread();
  }
  SpinLock m_lock;
  std::atomic<Int32> m_count = 0;
  AtomicInt32 m_atomic_add_count;
  AtomicInt32 m_atomic_sub_count;
  Int32 m_add_count;
  Int32 m_sub_count; 
 Int32 m_count2 = 0;
};

class RealTime
{
 public:
  static Real get()
  {
    return platform::getRealTime();
  }
};


class Test2
: public TraceAccessor
{
 public:

  explicit Test2(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

  void exec()
  {
    _exec1();
    _exec2();
  }

  void _exec1()
  {
    m_value = 0;
    Real v1 = RealTime::get();
    SpinLock slock;
    int n = 10000000;
    for( Integer i=0; i<n; ++i ){
      SpinLock::ScopedLock sl(slock);
      ++m_value;
    }
    Real v2 = RealTime::get();
      
    info() << "Value = " << m_value << " spin time=" << (v2-v1) / ((Real)n);
  }

  void _exec2()
  {
    m_value = 0;
    Real v1 = RealTime::get();
    Mutex slock;
    int n = 10000000;
    for( Integer i=0; i<n; ++i ){
      Mutex::ScopedLock sl(slock);
      ++m_value;
    }
    Real v2 = RealTime::get();
      
    info() << "Value = " << m_value << " mutex time=" << (v2-v1) / ((Real)n);
  }
 private:

  Int64 m_value = 0;
};

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du maillage
 */
class ThreadUnitTest
: public ArcaneThreadUnitTestObject
{
 public:

  explicit ThreadUnitTest(const ServiceBuildInfo& cb);
  ~ThreadUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_THREADUNITTEST(ThreadUnitTest,ThreadUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ThreadUnitTest::
ThreadUnitTest(const ServiceBuildInfo& mb)
: ArcaneThreadUnitTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ThreadUnitTest::
~ThreadUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ThreadUnitTest::
executeTest()
{
  {
    ThreadTest::Test1 t1(traceMng());
    t1.exec();
  }
  {
    ThreadTest::Test2 t2(traceMng());
    t2.exec();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ThreadUnitTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
