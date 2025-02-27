// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/Event.h"
#include "arcane/utils/FatalErrorException.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;


namespace
{
class TestMemberCall
{
 public:
  void my_func(int a,int b)
  {
    std::cout << "THIS_IS_MY FUNC XA=" << a << " B=" << b << '\n';
  }
  void operator()(int a,int b)
  {
    std::cout << "THIS_IS OPERATOR() FUNC XA=" << a << " B=" << b << '\n';
  }
};
}
TEST(TestEvent, Misc)
{
  using std::placeholders::_1;
  using std::placeholders::_2;

  int f = 3;
  auto func = [&](int a, int b) {
    std::cout << "XA=" << a << " B=" << b << " f=" << f << '\n';
    f = a + b;
  };
  auto func2 = [&](int a, int b) {
    std::cout << "FUNC2: XA=" << a << " B=" << b << " f=" << f << '\n';
  };
  TestMemberCall tmc;
  EventObserver<int, int> x2(func);
  {
    EventObservable<int, int> xevent;
    EventObserverPool pool;
    {
      EventObserver<int, int> xobserver;
      // NOTE: le test suivnant ne marche pas avec MSVS2013
      std::function<void(TestMemberCall*, int, int)> kk1(&TestMemberCall::my_func);
      std::function<void(int, int)> kk(std::bind(&TestMemberCall::my_func, tmc, _1, _2));
      //std::function<void(int,int)> kk2( std::bind( &TestMemberCall::my_func, tmc ) );
      //auto kk( std::bind( &TestMemberCall::my_func, &tmc ) );
      EventObserver<int, int> x4(kk);
      EventObserver<int, int> x3(tmc);
      xevent.attach(&x2);
      xevent.attach(&x3);
      xevent.attach(&x4);
      xevent.attach(&xobserver);
      xevent.notify(2, 3);
      xevent.detach(&x4);
    }
    xevent.attach(pool, func2);
  }
  std::cout << "(After) F=" << f << '\n';

  ASSERT_EQ(f, 5);

  {
    EventObserver<int, int>* eo1 = nullptr;
    EventObservable<int, int> xevent;
    {
      eo1 = new EventObserver<int, int>(std::bind(&TestMemberCall::my_func, tmc, _1, _2));
      xevent.attach(eo1);
    }
    xevent.notify(2, 4);
    delete eo1;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
