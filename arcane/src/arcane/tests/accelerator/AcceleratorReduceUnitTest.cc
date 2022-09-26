// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorReduceUnitTest.cc                                (C) 2000-2022 */
/*                                                                           */
/* Service de test des réductions sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"

#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/Reduce.h"
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
class AcceleratorReduceUnitTest
: public BasicUnitTest
{
 public:

  explicit AcceleratorReduceUnitTest(const ServiceBuildInfo& cb);
  ~AcceleratorReduceUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

 public:

  void _executeTest1();
  template<typename DataType> void _executeTestDataType();

  void _compareSum(Real reduced_sum,Real sum)
  {
    if (!math::isNearlyEqualWithEpsilon(reduced_sum,sum,1.0e-12))
      ARCANE_FATAL("Bad sum reduced_sum={0} expected={1}",reduced_sum,sum);
  }
  void _compareSum(Int64 reduced_sum,Int64 sum)
  {
    if (reduced_sum!=sum)
      ARCANE_FATAL("Bad sum reduced_sum={0} expected={1}",reduced_sum,sum);
  }
  void _compareSum(Int32 reduced_sum,Int32 sum)
  {
    if (reduced_sum!=sum)
      ARCANE_FATAL("Bad sum reduced_sum={0} expected={1}",reduced_sum,sum);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorReduceUnitTest, IUnitTest, AcceleratorReduceUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorReduceUnitTest::
AcceleratorReduceUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorReduceUnitTest::
~AcceleratorReduceUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorReduceUnitTest::
initializeTest()
{
  IApplication* app = subDomain()->application();
  const auto& acc_info = app->acceleratorRuntimeInitialisationInfo();
  initializeRunner(m_runner, traceMng(), acc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorReduceUnitTest::
executeTest()
{
  _executeTestDataType<Int64>();
  _executeTestDataType<Int32>();
  _executeTestDataType<double>();
}

template<typename DataType> void AcceleratorReduceUnitTest::
_executeTestDataType()
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Test1";

  auto queue = makeQueue(m_runner);

  constexpr int n1 = 3000000;

  NumArray<DataType, MDDim1> t1;
  t1.resize(n1);

  DataType sum = 0.0;
  DataType max_value = 0.0;
  DataType min_value = 0.0;
  for (Int32 i=0; i<n1; ++i ){
    int to_add = 2 + (rand() % 32);
    DataType v = static_cast<DataType>(to_add + ((i*2) % 257));

    sum += v;
    if (v>max_value || i==0)
      max_value = v;
    if (v<min_value || i==0)
      min_value = v;
    t1[i] = v;
  }
  info() << "SUM=" << sum << " MIN=" << min_value << " MAX=" << max_value;

  {
    auto command = makeCommand(queue);
    ax::ReducerSum<DataType> acc_sum(command);
    acc_sum.setValue(0.0);
    auto in_t1 = viewIn(command, t1);

    command << RUNCOMMAND_LOOP1(iter, n1)
    {
      DataType v = in_t1(iter);
      acc_sum.add(v);
    };

    DataType reduced_sum = acc_sum.reduce();
    info() << "REDUCED_SUM=" << reduced_sum;
    _compareSum(reduced_sum,sum);
  }

  {
    auto command = makeCommand(queue);
    ax::ReducerMin<DataType> acc_min(command);
    auto in_t1 = viewIn(command, t1);

    command << RUNCOMMAND_LOOP1(iter, n1)
    {
      DataType v = in_t1(iter);
      acc_min.min(v);
    };

    DataType reduced_min = acc_min.reduce();
    info() << "REDUCED_MIN=" << reduced_min;
    if (reduced_min!=min_value)
      ARCANE_FATAL("Bad minimum reduced_min={0} expected={1}",reduced_min,min_value);
  }

  {
    auto command = makeCommand(queue);
    ax::ReducerMax<DataType> acc_max(command);
    auto in_t1 = viewIn(command, t1);

    command << RUNCOMMAND_LOOP1(iter, n1)
    {
      DataType v = in_t1(iter);
      acc_max.max(v);
    };

    DataType reduced_max = acc_max.reduce();
    info() << "REDUCED_MAX=" << reduced_max;
    if (reduced_max!=max_value)
      ARCANE_FATAL("Bad minimum reduced_min={0} expected={1}",reduced_max,max_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
