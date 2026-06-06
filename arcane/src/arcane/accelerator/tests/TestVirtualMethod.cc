// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumArray.h"
#include "arcane/utils/Exception.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include "TestVirtualMethod.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void arcaneRegisterDefaultAcceleratorRuntime();
extern "C++" Arcane::Accelerator::eExecutionPolicy arcaneGetDefaultExecutionPolicy();

using namespace Arcane;
using namespace Arcane::Accelerator;

namespace
{
void _doInit()
{
  arcaneRegisterDefaultAcceleratorRuntime();
}
Arcane::Accelerator::eExecutionPolicy _defaultExecutionPolicy()
{
  return arcaneGetDefaultExecutionPolicy();
}
} // namespace

extern "C++" void
_doCallTestVirtualMethod1(RunQueue& queue, NumArray<Int32, MDDim1>& compute_array, BaseTestClass* base_instance);

class DerivedTestClass
: public BaseTestClass
{
 public:

  ARCCORE_HOST_DEVICE int apply(int a, int b) override
  {
    return a + b;
  }
};

void _doTestVirtualMethod1(eExecutionPolicy exec_policy)
{
  std::cout << "Test Virtual Method 1. Execution policy=" << exec_policy << "\n";
  Runner runner(exec_policy);

  RunQueue queue(makeQueue(runner));
  eMemoryResource mem_resource = eMemoryResource::Host;
  if (queue.isAcceleratorPolicy())
    mem_resource = eMemoryResource::Device;

  // Create an instance of 'DerivedTestClass' in the accelerator memory.
  // To do this, you must allocate memory on the accelerator and call
  // the constructor of 'DerivedTestClass' on the accelerator so that
  // the virtual method table is correctly initialized.
  NumArray<Byte, MDDim1> instance_memory(mem_resource);
  instance_memory.resize(sizeof(DerivedTestClass));

  BaseTestClass* base_instance = reinterpret_cast<DerivedTestClass*>(instance_memory.bytes().data());
  std::cout << "Test Virtual Method 1. Create derived class\n";
  std::cout.flush();

  {
    RunCommand command(makeCommand(queue));
    command << RUNCOMMAND_SINGLE()
    {
      new (base_instance) DerivedTestClass();
    };
  }

  // Applies a command that takes the base class pointer as an argument.
  const Int32 nb_item = 12;
  NumArray<Int32, MDDim1> compute_array(mem_resource);
  compute_array.resize(nb_item);
  std::cout << "Test Virtual Method 1. Do computation\n";
  std::cout.flush();
  _doCallTestVirtualMethod1(queue, compute_array, base_instance);

  NumArray<Int32, MDDim1> host_array;
  host_array.copy(compute_array);

  for (Int32 i = 0; i < nb_item; ++i)
    std::cout << "I=" << i << " R=" << host_array[i] << "\n";
  for (Int32 i = 0; i < nb_item; ++i)
    ASSERT_EQ(i * 2, host_array[i]);
}

TEST(ArcaneAccelerator, VirtualMethod)
{
  _doInit();

  auto f = [] {
    _doTestVirtualMethod1(_defaultExecutionPolicy());
    _doTestVirtualMethod1(eExecutionPolicy::Sequential);
  };
  return arcaneCallFunctionAndTerminateIfThrow(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
