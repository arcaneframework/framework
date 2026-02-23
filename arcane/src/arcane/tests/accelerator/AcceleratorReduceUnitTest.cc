// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorReduceUnitTest.cc                                (C) 2000-2026 */
/*                                                                           */
/* Service de test des réductions sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/GenericReducer.h"

#include "arcane/tests/accelerator/AcceleratorReduceUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
using namespace Arcane::Accelerator;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'NumArray'.
 */
class AcceleratorReduceUnitTest
: public ArcaneAcceleratorReduceUnitTestObject
{
 public:

  explicit AcceleratorReduceUnitTest(const ServiceBuildInfo& cb);
  ~AcceleratorReduceUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  Runner m_runner;
  RunQueue m_queue;

 public:

  void _executeTest1();
  template <typename DataType> void _executeTestDataType(Int32 nb_iteration);

  void _compareSum(Real reduced_sum, Real sum)
  {
    if (!math::isNearlyEqualWithEpsilon(reduced_sum, sum, 1.0e-12))
      ARCANE_FATAL("Bad sum reduced_sum={0} expected={1}", reduced_sum, sum);
  }
  void _compareSum(Int64 reduced_sum, Int64 sum)
  {
    if (reduced_sum != sum)
      ARCANE_FATAL("Bad sum reduced_sum={0} expected={1}", reduced_sum, sum);
  }
  void _compareSum(Int32 reduced_sum, Int32 sum)
  {
    if (reduced_sum != sum)
      ARCANE_FATAL("Bad sum reduced_sum={0} expected={1}", reduced_sum, sum);
  }

  template <typename DataType> void
  _executeTestReduceSum(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1, DataType expected_value);
  template <typename DataType> void
  _executeTestReduceMin(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1, DataType expected_value);
  template <typename DataType> void
  _executeTestReduceMax(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1, DataType expected_value);
  template <typename DataType> void
  _executeTestReduceDirect(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1,
                           DataType expected_sum,
                           DataType expected_min,
                           DataType expected_max);
  template <typename DataType> void
  _executeTestReduceWithIndex(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1,
                              DataType expected_sum,
                              DataType expected_min,
                              DataType expected_max);

  template <typename DataType> void
  _executeTestReduceV2(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1,
                       DataType expected_sum,
                       DataType expected_min,
                       DataType expected_max);

 private:

  void executeTest2(Int32 nb_iteration);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorReduceUnitTest, IUnitTest, AcceleratorReduceUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorReduceUnitTest::
AcceleratorReduceUnitTest(const ServiceBuildInfo& sb)
: ArcaneAcceleratorReduceUnitTestObject(sb)
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
  IAcceleratorMng* amng = subDomain()->acceleratorMng();
  m_runner = *amng->defaultRunner();

  auto queue = makeQueue(m_runner);
  m_queue = queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorReduceUnitTest::
executeTest()
{
  info() << "ExecuteReduceTest policy=" << m_queue.executionPolicy();
  info() << "UseReducePolicy = Grid";
  m_runner.setDeviceReducePolicy(ax::eDeviceReducePolicy::Grid);
  Int32 nb_iter = 100;
  if (!isAcceleratorPolicy(m_runner.executionPolicy()))
    nb_iter = 10;
  if (arcaneIsDebug())
    nb_iter /= 5;
  executeTest2(nb_iter);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorReduceUnitTest::
executeTest2(Int32 nb_iteration)
{
  _executeTestDataType<Int64>(nb_iteration);
  _executeTestDataType<Int32>(nb_iteration);
  _executeTestDataType<double>(nb_iteration);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorReduceUnitTest::
_executeTestDataType(Int32 nb_iteration)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Test1 nb_iter=" << nb_iteration << " policy=" << m_queue.executionPolicy();

  constexpr int n1 = 3000000;

  NumArray<DataType, MDDim1> t1;
  t1.resize(n1);
  ConstMemoryView t1_mem_view(makeMemoryView(t1.to1DSpan()));
  m_runner.setMemoryAdvice(t1_mem_view, ax::eMemoryAdvice::PreferredLocationDevice);
  m_runner.setMemoryAdvice(t1_mem_view, ax::eMemoryAdvice::AccessedByHost);
  DataType sum = 0.0;
  DataType max_value = 0.0;
  DataType min_value = 0.0;
  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rand() % 32);
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 257));

    sum += v;
    if (v > max_value || i == 0)
      max_value = v;
    if (v < min_value || i == 0)
      min_value = v;
    t1[i] = v;
  }
  info() << "SUM=" << sum << " MIN=" << min_value << " MAX=" << max_value;

  //NumArray<DataType, MDDim1> t1; //(eMemoryRessource::Device);
  //t1.copy(host_t1);
  m_runner.setMemoryAdvice(t1_mem_view, ax::eMemoryAdvice::MostlyRead);
  m_queue.prefetchMemory(ax::MemoryPrefetchArgs(t1_mem_view).addAsync());

  _executeTestReduceV2(nb_iteration, t1, sum, min_value, max_value);

  // Les tests suivants avec les réductions historiques ne sont pas supportés en SYCL
  if (m_queue.executionPolicy() != eExecutionPolicy::SYCL) {
    info() << "Check reduction V1 (sync)";
    _executeTestReduceSum(nb_iteration, t1, sum);
    _executeTestReduceMin(nb_iteration, t1, min_value);
    _executeTestReduceMax(nb_iteration, t1, max_value);

    // Test aussi en mode asynchrone
    m_queue.setAsync(true);
    info() << "Check reduction V1 (async)";
    _executeTestReduceSum(nb_iteration, t1, sum);
    _executeTestReduceMin(nb_iteration, t1, min_value);
    _executeTestReduceMax(nb_iteration, t1, max_value);
    m_queue.setAsync(false);
}

  // Utilisation des kernels spécifiques
  _executeTestReduceDirect(nb_iteration, t1, sum, min_value, max_value);

  // Utilisation des kernels spécifiques avec index
  _executeTestReduceWithIndex(nb_iteration, t1, sum, min_value, max_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorReduceUnitTest::
_executeTestReduceDirect(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1,
                         DataType expected_sum,
                         DataType expected_min,
                         DataType expected_max)
{
  // Utilisation des kernels spécifiques
  for (int z = 0; z < nb_iteration; ++z) {
    ax::GenericReducer<DataType> reducer(m_queue);
    reducer.applySum(t1, A_FUNCINFO);
    DataType reduced_sum = reducer.reducedValue();
    if (z == 0)
      info() << "REDUCED_SUM (direct)=" << reduced_sum;
    _compareSum(reduced_sum, expected_sum);
  }

  for (int z = 0; z < nb_iteration; ++z) {
    ax::GenericReducer<DataType> reducer(m_queue);
    reducer.applyMin(t1, A_FUNCINFO);
    DataType reduced_min = reducer.reducedValue();
    if (z == 0)
      info() << "REDUCED_MIN (direct)=" << reduced_min;
    if (reduced_min != expected_min)
      ARCANE_FATAL("Bad minimum reduced_min={0} expected={1}", reduced_min, expected_min);
  }

  for (int z = 0; z < nb_iteration; ++z) {
    ax::GenericReducer<DataType> reducer(m_queue);
    reducer.applyMax(t1, A_FUNCINFO);
    DataType reduced_max = reducer.reducedValue();
    if (z == 0)
      info() << "REDUCED_MAX (direct)=" << reduced_max;
    if (reduced_max != expected_max)
      ARCANE_FATAL("Bad maximum reduced_max={0} expected={1}", reduced_max, expected_max);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorReduceUnitTest::
_executeTestReduceWithIndex(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1,
                            DataType expected_sum,
                            DataType expected_min,
                            DataType expected_max)
{
  auto t1_view = t1.to1DSmallSpan();
  const Int32 n1 = t1_view.size();
  auto getter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> DataType {
    return t1_view[index];
  };

  for (int z = 0; z < nb_iteration; ++z) {
    ax::GenericReducer<DataType> reducer(m_queue);
    reducer.applySumWithIndex(n1, getter_lambda, A_FUNCINFO);
    DataType reduced_sum = reducer.reducedValue();
    if (z == 0)
      info() << "REDUCED_SUM (direct with index)=" << reduced_sum;
    _compareSum(reduced_sum, expected_sum);
  }

  for (int z = 0; z < nb_iteration; ++z) {
    ax::GenericReducer<DataType> reducer(m_queue);
    reducer.applyMinWithIndex(n1, getter_lambda, A_FUNCINFO);
    DataType reduced_min = reducer.reducedValue();
    if (z == 0)
      info() << "REDUCED_MIN (direct with index)=" << reduced_min;
    if (reduced_min != expected_min)
      ARCANE_FATAL("Bad minimum reduced_min={0} expected={1}", reduced_min, expected_min);
  }

  for (int z = 0; z < nb_iteration; ++z) {
    ax::GenericReducer<DataType> reducer(m_queue);
    reducer.applyMaxWithIndex(n1, getter_lambda, A_FUNCINFO);
    DataType reduced_max = reducer.reducedValue();
    if (z == 0)
      info() << "REDUCED_MAX (direct with index)=" << reduced_max;
    if (reduced_max != expected_max)
      ARCANE_FATAL("Bad maximum reduced_max={0} expected={1}", reduced_max, expected_max);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorReduceUnitTest::
_executeTestReduceSum(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1, DataType expected_value)
{
  const Int32 n1 = t1.extent0();
  for (int z = 0; z < nb_iteration; ++z) {
    auto command = makeCommand(m_queue);
    ax::ReducerSum<DataType> acc_sum(command);
    acc_sum.setValue(0.0);
    auto in_t1 = viewIn(command, t1);

    command << RUNCOMMAND_LOOP1(iter, n1)
    {
      DataType v = in_t1(iter);
      acc_sum.add(v);
    };

    DataType reduced_sum = acc_sum.reduce();
    if (z == 0)
      info() << "REDUCED_SUM=" << reduced_sum;
    _compareSum(reduced_sum, expected_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorReduceUnitTest::
_executeTestReduceMin(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1, DataType expected_value)
{
  const Int32 n1 = t1.extent0();
  for (int z = 0; z < nb_iteration; ++z) {
    auto command = makeCommand(m_queue);
    ax::ReducerMin<DataType> acc_min(command);
    auto in_t1 = viewIn(command, t1);

    command << RUNCOMMAND_LOOP1(iter, n1)
    {
      DataType v = in_t1(iter);
      acc_min.min(v);
    };

    DataType reduced_min = acc_min.reduce();
    if (z == 0)
      info() << "REDUCED_MIN=" << reduced_min;
    if (reduced_min != expected_value)
      ARCANE_FATAL("Bad minimum reduced_min={0} expected={1}", reduced_min, expected_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorReduceUnitTest::
_executeTestReduceMax(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1, DataType expected_value)
{
  const Int32 n1 = t1.extent0();
  for (int z = 0; z < nb_iteration; ++z) {
    auto command = makeCommand(m_queue);
    ax::ReducerMax<DataType> acc_max(command);
    auto in_t1 = viewIn(command, t1);

    command << RUNCOMMAND_LOOP1(iter, n1)
    {
      DataType v = in_t1(iter);
      acc_max.max(v);
    };

    DataType reduced_max = acc_max.reduce();
    if (z == 0)
      info() << "REDUCED_MAX=" << reduced_max;
    if (reduced_max != expected_value)
      ARCANE_FATAL("Bad maximum reduced_max={0} expected={1}", reduced_max, expected_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorReduceUnitTest::
_executeTestReduceV2(Int32 nb_iteration, const NumArray<DataType, MDDim1>& t1,
                     DataType expected_sum,
                     DataType expected_min,
                     DataType expected_max)
{
  info() << "Execute Test ReduceV2 nb_iter=" << nb_iteration;
  const Int32 n1 = t1.extent0();
  for (int z = 0; z < nb_iteration; ++z) {
    auto command = makeCommand(m_queue);
    ReducerSum2<DataType> reducer_sum(command);
    ReducerMin2<DataType> reducer_min(command);
    ReducerMax2<DataType> reducer_max(command);

    auto in_t1 = viewIn(command, t1);
    SimpleForLoopRanges<1> range(n1);
    command << RUNCOMMAND_LOOP(iter, range, reducer_sum, reducer_max, reducer_min)
    {
      DataType v = in_t1(iter);
      reducer_sum.combine(v);
      reducer_min.combine(v);
      reducer_max.combine(v);
    };

    DataType reduced_max = reducer_max.reducedValue();
    DataType reduced_min = reducer_min.reducedValue();
    DataType reduced_sum = reducer_sum.reducedValue();
    if (z == 0) {
      info() << "REDUCEDV2_MAX=" << reduced_max;
      info() << "REDUCEDV2_MIN=" << reduced_min;
      info() << "REDUCEDV2_SUM=" << reduced_sum;
    }
    if (reduced_max != expected_max)
      ARCANE_FATAL("Bad reduced_max={0} expected={1}", reduced_max, expected_max);
    if (reduced_min != expected_min)
      ARCANE_FATAL("Bad reduced_min={0} expected={1}", reduced_min, expected_min);
    if (reduced_sum != expected_sum)
      ARCANE_FATAL("Bad reduced_sum={0} expected={1}", reduced_sum, expected_sum);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
