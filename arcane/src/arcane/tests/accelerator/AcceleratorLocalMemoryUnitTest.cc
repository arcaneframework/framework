// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorLocalMemoryUnitTest.cc                           (C) 2000-2025 */
/*                                                                           */
/* Service de test de la mémoire locale pour les accélérateurs.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"

#include "arcane/utils/NumArray.h"
#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/NumArrayViews.h"

#include "arcane/accelerator/RunCommandLocalMemory.h"
#include "arcane/accelerator/Atomic.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'AcceleratorViews'.
 */
class AcceleratorLocalMemoryUnitTest
: public BasicUnitTest
{
 public:

  explicit AcceleratorLocalMemoryUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

 public:

  void _executeTest1();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorLocalMemoryUnitTest, IUnitTest,
                                           AcceleratorLocalMemoryUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorLocalMemoryUnitTest::
AcceleratorLocalMemoryUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorLocalMemoryUnitTest::
initializeTest()
{
  m_runner = *(subDomain()->acceleratorMng()->defaultRunner());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorLocalMemoryUnitTest::
executeTest()
{
  _executeTest1();
}

#if defined(ARCCORE_DEVICE_CODE)
#define RUNCOMMAND_SYNCTHREADS() __syncthreads();
#else
#define RUNCOMMAND_SYNCTHREADS()
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorLocalMemoryUnitTest::
_executeTest1()
{
  auto queue = makeQueue(m_runner);
  // Pour le moment cela ne fonctionne que sur accélérateur CUDA ou HIP
  {
    Int32 loop_size = 1024 * 1024;
    info() << "DO_LOOP2 LocalMemory size=" << loop_size;
    auto command = makeCommand(queue);
    ax::RunCommandLocalMemory<Int32> local_data(command, 50);
    NumArray<Int32, MDDim1> out_array(loop_size);
    out_array.fillHost(0);
    auto out_span = viewInOut(command, out_array);
    command << RUNCOMMAND_LOOP1(iter, loop_size, local_data)
    {
      auto [i] = iter();
      auto s = local_data.span();
      if (i == 0)
        for (int j = 0; j < 50; ++j)
          s[j] = 0;
      RUNCOMMAND_SYNCTHREADS();
      ax::doAtomic<ax::eAtomicOperation::Add>(&s[i % 50], 1);
      //++s[i % 50];
      RUNCOMMAND_SYNCTHREADS();
      if (i == 0) {
        for (int j = 0; j < 50; ++j)
          out_span[i] += s[j];
      }
    };

    Int32 out_value = out_span[0];
    info() << "DO_LOOP2 LocalMemory out[0]=" << out_value;
    if (out_value != 256)
      ARCANE_FATAL("Bad value expected=256 v={0}", out_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
