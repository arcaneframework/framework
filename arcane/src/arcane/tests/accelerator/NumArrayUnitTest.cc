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
  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);

  constexpr int n1 = 1000;
  constexpr int n2 = 3;
  constexpr int n3 = 4;
  constexpr int n4 = 13;

  // TODO: vérifier le calcul.

  {
    NumArray<double,1> t1(n1);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP1(iter,n1)
    {
      auto [i] = iter();
      out_t1(i) = static_cast<double>(i*2);
    };
  }

  {
    NumArray<double,2> t1(n1,n2);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP2(iter,n1,n2)
    {
      auto [i, j] = iter();
      out_t1(i,j) = static_cast<double>(i*2 + j*3);
    };
  }

  {
    NumArray<double,3> t1(n1,n2,n3);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP3(iter,n1,n2,n3)
    {
      auto [i, j, k] = iter();
      out_t1(i,j,k) = static_cast<double>(i*2 + j*3 + k*4);
    };
  }

  {
    NumArray<double,4> t1(n1,n2,n3,n4);

    auto out_t1 = ax::viewOut(command,t1);

    command << RUNCOMMAND_LOOP4(iter,n1,n2,n3,n4)
    {
      auto [i, j, k, l] = iter();
      out_t1(i,j,k,l) = static_cast<double>(i*2 + j*3 + k*4 + l*8);
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
