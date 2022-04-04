// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ThreadUnitTest.cc                                           (C) 2000-2010 */
/*                                                                           */
/* Service de test des threads.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/SpinLock.h"
#include "arcane/utils/Mutex.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/ThreadUnitTest_axl.h"

#include "arcane/utils/Functor.h"
#include "arcane/utils/IThreadImplementation.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Atomic.h"

#include <glib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

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
  MyThread() : m_functor(0), m_thread(0) {}
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
class Test1 : public TraceAccessor
{
 public:
  Test1(ITraceMng* tm) : TraceAccessor(tm), m_count(0), m_count2(0){}
 public:
  void exec()
  {
    for( Integer i=0; i<100; ++i ){
      FunctorT<Test1> f1(this,&Test1::_F1);
      UniqueArray<MyThread> threads(10);
      for( Integer j=0; j<10; ++j )
        threads[j].create(&f1);
      for( Integer j=0; j<10; ++j )
        threads[j].join();
    }
    if (m_count.value()!=1000)
      fatal() << "Bad value for atomic count: v=" << m_count.value() << " expected=1000";
    if (m_count2!=10000)
      fatal() << "Bad value for count2: v=" << m_count.value() << " expected=10000";
  }
  void _F1()
  {
    ++m_count;
    {
      SpinLock::ScopedLock sl(m_lock);
      m_count2 += 10;
    }
  }
  SpinLock m_lock;
  AtomicInt32 m_count;
  Int32 m_count2;
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

  Test2(ITraceMng* tm) : TraceAccessor(tm), m_value(0)
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

  Int64 m_value;
};

class Test3
: public TraceAccessor
{
 public:

  Test3(ITraceMng* tm) : TraceAccessor(tm)
  {
  }

 public:

  void exec()
  {
    _exec1();
  }

  void _exec1()
  {
    Real v1 = RealTime::get();
    int n = 10000000;
    GPrivate* prv = g_private_new(0);
    Int64 my_value = 1;
    g_private_set(prv,&my_value);
    Int64 total = 0;
    for( Integer i=0; i<n; ++i ){
      Int64* v = (Int64*)g_private_get(prv);
      total += *v;
    }
    Real v2 = RealTime::get();
      
    info() << "Value = " << total << " private_key=" << (v2-v1) / ((Real)n);
  }

 private:

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

public:

  ThreadUnitTest(const ServiceBuildInfo& cb);
  ~ThreadUnitTest();

 public:

  virtual void initializeTest();
  virtual void executeTest();

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
  {
    ThreadTest::Test3 t3(traceMng());
    t3.exec();
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

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
