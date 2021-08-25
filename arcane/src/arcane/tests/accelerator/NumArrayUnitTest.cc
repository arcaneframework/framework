// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayUnitTest.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Service de test des 'NumArray'.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"

#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'NumArray'.
 */
class NumArrayUnitTest
: public BasicUnitTest
{
 public:

  explicit NumArrayUnitTest(const ServiceBuildInfo& cb);
  ~NumArrayUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

  static constexpr double _getValue(Int64 i)
  {
    return static_cast<double>(i*2);
  }
  static constexpr double _getValue(Int64 i,Int64 j)
  {
    return static_cast<double>(i*2 + j*3);
  }
  static constexpr double _getValue(Int64 i,Int64 j,Int64 k)
  {
    return static_cast<double>(i*2 + j*3 + k*4);
  }
  static constexpr double _getValue(Int64 i,Int64 j,Int64 k,Int64 l)
  {
    return static_cast<double>(i*2 + j*3 + k*4 + l*8);
  }

  template<int Rank> double
  _doSum(NumArray<double,Rank> values,ArrayBounds<Rank> bounds)
  {
    double total = 0.0;
    Accelerator::impl::applyGenericLoopSequential(bounds,[&](ArrayBoundsIndex<Rank> idx){ total += values(idx); });
    return total;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(NumArrayUnitTest,IUnitTest,NumArrayUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NumArrayUnitTest::
NumArrayUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NumArrayUnitTest::
~NumArrayUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
initializeTest()
{
  IApplication* app = subDomain()->application();
  const auto& acc_info = app->acceleratorRuntimeInitialisationInfo();
  initializeRunner(m_runner,traceMng(),acc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
executeTest()
{
  ValueChecker vc(A_FUNCINFO);

  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);

  // Ne pas changer les dimensions du tableau sinon
  // il faut aussi changer le calcul des sommes
  constexpr int n1 = 1000;
  constexpr int n2 = 3;
  constexpr int n3 = 4;
  constexpr int n4 = 13;

  constexpr double expected_sum1 = 999000.0;
  constexpr double expected_sum2 = 3006000.0;
  constexpr double expected_sum3 = 12096000.0;
  constexpr double expected_sum4 = 164736000.0;

  // TODO: vérifier le calcul.

  {
    NumArray<double,1> t1(n1);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP1(iter,n1)
    {
      auto [i] = iter();
      out_t1(i) = _getValue(i);
    };
    double s1 = _doSum(t1,ArrayBounds<1>(n1));
    info() << "SUM1 = " << s1;
    vc.areEqual(s1,expected_sum1,"SUM1");
  }

  {
    NumArray<double,2> t1(n1,n2);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP2(iter,n1,n2)
    {
      auto [i, j] = iter();
      out_t1(i,j) = _getValue(i,j);
    };
    double s2 = _doSum(t1,{n1,n2});
    info() << "SUM2 = " << s2;
    vc.areEqual(s2,expected_sum2,"SUM2");
  }

  {
    NumArray<double,3> t1(n1,n2,n3);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP3(iter,n1,n2,n3)
    {
      auto [i, j, k] = iter();
      out_t1(i,j,k) = _getValue(i,j,k);
    };
    double s3 = _doSum(t1,{n1,n2,n3});
    info() << "SUM3 = " << s3;
    vc.areEqual(s3,expected_sum3,"SUM3");
  }

  {
    NumArray<double,4> t1(n1,n2,n3,n4);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP4(iter,n1,n2,n3,n4)
    {
      auto [i, j, k, l] = iter();
      out_t1(i,j,k,l) = _getValue(i,j,k,l);
    };
    double s4 = _doSum(t1,{n1,n2,n3,n4});
    info() << "SUM4 = " << s4;
    vc.areEqual(s4,expected_sum4,"SUM4");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
