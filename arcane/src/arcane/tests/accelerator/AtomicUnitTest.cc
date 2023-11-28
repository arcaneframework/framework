// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AtomicUnitTest.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Service de test des fonctions 'atomic' sur accélérateur.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/IMemoryRessourceMng.h"

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

  template<typename DataType> void _executeTest1(eMemoryRessource mem_ressource);
  template<typename DataType> void _executeTestType();
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
  _executeTestType<Real>();
  _executeTestType<Int32>();
  _executeTestType<Int64>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void AtomicUnitTest::
_executeTestType()
{
  ax::eExecutionPolicy policy = m_runner.executionPolicy();
  if (ax::impl::isAcceleratorPolicy(policy)) {
    info() << "ExecuteTest1: using accelerator";
    if (policy!=ax::eExecutionPolicy::HIP){
      // Ne fonctionne pas sur les gros tableaux (>50000 valeurs) avec ROCM
      _executeTest1<DataType>(eMemoryRessource::UnifiedMemory);
      _executeTest1<DataType>(eMemoryRessource::HostPinned);
    }
    _executeTest1<DataType>(eMemoryRessource::Device);
  }
  else {
    info() << "ExecuteTest1: using host";
    _executeTest1<DataType>(eMemoryRessource::Host);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void AtomicUnitTest::
_executeTest1(eMemoryRessource mem_ressource)
{
  info() << "Test Atomic";
  const Int32 nb_value = 83239;
  //const Int32 nb_value = 1250;
  NumArray<DataType, MDDim1> v0(nb_value);
  DataType ref_value = 0;
  const DataType add0 = static_cast<DataType>(2);
  for (Int32 i = 0; i < nb_value; ++i) {
    DataType x = static_cast<DataType>(i % (nb_value/4));
    v0[i] = x;
    DataType to_add = v0[i] + x + add0;
    ref_value += to_add;
  }

  auto queue = makeQueue(m_runner);
  NumArray<DataType, MDDim1> v_sum(1, mem_ressource);
  v_sum.fill(0.0, &queue);
  DataType* device_sum_ptr = &v_sum[0];
  {
    auto command = makeCommand(queue);
    auto inout_a = viewInOut(command, v0);

    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      DataType x = static_cast<DataType>(i % (nb_value/4));
      DataType v = x + add0;
      ax::atomicAdd(inout_a(iter), v);
      ax::atomicAdd(device_sum_ptr, inout_a(iter));
    };
  }

  DataType sum = {};
  for (Int32 i = 0; i < nb_value; ++i) {
    sum += v0[i];
  }
  NumArray<DataType, MDDim1> host_sum(1);
  host_sum.copy(v_sum);
  info() << "SUM=" << sum;
  info() << "REF=" << ref_value;
  info() << "V_SUM=" << host_sum[0];
  if (sum != ref_value)
    ARCANE_FATAL("Bad value sum={0} expected={1}", sum, ref_value);
  if (host_sum[0] != ref_value)
    ARCANE_FATAL("Bad value host_sum={0} expected={1}", host_sum[0], ref_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
