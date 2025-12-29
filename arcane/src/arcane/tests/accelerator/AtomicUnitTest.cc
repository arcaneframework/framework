// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AtomicUnitTest.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Service de test des fonctions 'atomic' sur accélérateur.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/PointerAttribute.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/Runner.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/SpanViews.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Atomic.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
}

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'NumArray'.
 */
class AtomicUnitTest
: public BasicUnitTest
{
 public:

  explicit AtomicUnitTest(const ServiceBuildInfo& cb);
  ~AtomicUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

 public:

  template <typename DataType, enum ax::eAtomicOperation Operation>
  void _executeTest1(eMemoryRessource mem_ressource);
  template <enum ax::eAtomicOperation Operation>
  void _executeTestOperation();
  template <typename DataType, enum ax::eAtomicOperation Operation>
  void _executeTestType();
  void executeTestSample();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AtomicUnitTest, IUnitTest, AcceleratorAtomicUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AtomicUnitTest::
AtomicUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AtomicUnitTest::
~AtomicUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AtomicUnitTest::
initializeTest()
{
  Runner* m = subDomain()->acceleratorMng()->defaultRunner();
  ARCANE_CHECK_POINTER(m);
  m_runner = *m;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AtomicUnitTest::
executeTest()
{
  _executeTestOperation<ax::eAtomicOperation::Add>();
  _executeTestOperation<ax::eAtomicOperation::Max>();
  _executeTestOperation<ax::eAtomicOperation::Min>();
  executeTestSample();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AtomicUnitTest::
executeTestSample()
{
  //![SampleAtomicAdd]
  using namespace Arcane;
  namespace ax = Arcane::Accelerator;
  Arcane::Accelerator::RunQueue queue = makeQueue(m_runner);
  Arcane::NumArray<Arcane::Real, MDDim1> v_sum(100);
  auto command = makeCommand(queue);
  auto inout_a = viewInOut(command, v_sum);
  Real v_to_add = 2.1;
  constexpr auto Add = ax::eAtomicOperation::Add;
  command << RUNCOMMAND_LOOP1(iter, 100)
  {
    // atomic add 'v' to 'inout_a(iter)'
    ax::doAtomic<Add>(inout_a(iter), v_to_add);
  };
  //![SampleAtomicAdd]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <enum ax::eAtomicOperation Operation> void AtomicUnitTest::
_executeTestOperation()
{
  _executeTestType<Real, Operation>();
  _executeTestType<Int32, Operation>();
  _executeTestType<Int64, Operation>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, enum ax::eAtomicOperation Operation>
void AtomicUnitTest::
_executeTestType()
{
  ax::eExecutionPolicy policy = m_runner.executionPolicy();
  if (isAcceleratorPolicy(policy)) {
    info() << "ExecuteTest1: using accelerator";
    if (policy != ax::eExecutionPolicy::HIP) {
      // Ne fonctionne pas sur les gros tableaux (>50000 valeurs) avec ROCM
      _executeTest1<DataType, Operation>(eMemoryRessource::UnifiedMemory);
      _executeTest1<DataType, Operation>(eMemoryRessource::HostPinned);
    }
    _executeTest1<DataType, Operation>(eMemoryRessource::Device);
  }
  else {
    info() << "ExecuteTest1: using host";
    _executeTest1<DataType, Operation>(eMemoryRessource::Host);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, enum ax::eAtomicOperation Operation>
void AtomicUnitTest::
_executeTest1(eMemoryRessource mem_ressource)
{
  info() << "Test Atomic ressource=" << mem_ressource << " Operation=" << (int)Operation;
  const Int32 nb_value = 83239;
  //const Int32 nb_value = 1250;
  NumArray<DataType, MDDim1> v0(nb_value);
  DataType ref_value = 0;
  const DataType add0 = static_cast<DataType>(2);
  DataType init_value = {};
  for (Int32 i = 0; i < nb_value; ++i) {
    DataType x = static_cast<DataType>(i % (nb_value / 4));
    if ((i % 2) == 0)
      x = -x;
    v0[i] = x;
    DataType value_to_apply = x + add0;
    switch (Operation) {
    case ax::eAtomicOperation::Add:
      ref_value += v0[i] + value_to_apply;
      break;
    case ax::eAtomicOperation::Min:
      if (i == 0)
        ref_value = init_value = value_to_apply;
      else if (value_to_apply < ref_value)
        ref_value = value_to_apply;
      v0[i] = ref_value;
      break;
    case ax::eAtomicOperation::Max:
      if (i == 0)
        ref_value = init_value = value_to_apply;
      else if (ref_value < value_to_apply)
        ref_value = value_to_apply;
      v0[i] = ref_value;
      break;
    }
    if (i < 10 || i > (nb_value - 10))
      info() << "I=" << i << " ref_value=" << ref_value << " to_apply=" << value_to_apply;
  }

  auto queue = makeQueue(m_runner);
  NumArray<DataType, MDDim1> v_sum(1, mem_ressource);
  NumArray<bool, MDDim1> is_ok_array(nb_value);
  v_sum.fill(init_value, &queue);
  DataType* device_sum_ptr = &v_sum[0];
  {
    auto command = makeCommand(queue);
    auto inout_a = viewInOut(command, v0);
    auto out_is_ok = viewOut(command, is_ok_array);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      DataType x = static_cast<DataType>(i % (nb_value / 4));
      if ((i % 2) == 0)
        x = -x;
      DataType v = x + add0;
      DataType old_v = ax::doAtomic<Operation>(inout_a(iter), v);
      DataType new_v = inout_a(iter);
      // Si l'opération est l'ajout, teste que l'ancienne valeur plus
      // la valeur ajoutée vaut la nouvelle
      if (Operation == ax::eAtomicOperation::Add) {
        out_is_ok[i] = (new_v == (old_v + v));
      }
      else
        out_is_ok[i] = true;
      ax::doAtomic<Operation>(device_sum_ptr, inout_a(iter));
    };
  }

  DataType cumulative = init_value;
  for (Int32 i = 0; i < nb_value; ++i) {
    if (i < 10)
      info() << "V[" << i << "] = " << v0[i] << " is_ok=" << is_ok_array[i];
    ax::doAtomic<Operation>(&cumulative, v0[i]);
    if (!is_ok_array[i])
      ARCANE_FATAL("Bad old value for index '{0}'", i);
  }
  NumArray<DataType, MDDim1> host_cumulative(1);
  host_cumulative.copy(v_sum);
  info() << "CURRENT=" << cumulative;
  info() << "REF=" << ref_value;
  info() << "V_CURRENT=" << host_cumulative[0];
  if (cumulative != ref_value)
    ARCANE_FATAL("Bad value cumulative={0} expected={1}", cumulative, ref_value);
  if (host_cumulative[0] != ref_value)
    ARCANE_FATAL("Bad value host_cumulative={0} expected={1}", host_cumulative[0], ref_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
