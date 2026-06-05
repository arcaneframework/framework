// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorLocalMemoryUnitTest.cc                           (C) 2000-2026 */
/*                                                                           */
/* Local memory test service for accelerators.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ValueChecker.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/NumArray.h"
#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/LocalMemory.h"
#include "arcane/accelerator/Atomic.h"
#include "arcane/accelerator/RunCommandLaunch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Brief test service for the 'AcceleratorViews' class.
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
  RunQueue m_queue;

 public:

  void _executeTest1();
  void _doTest(Int32 group_size, Int32 nb_group_or_total_nb_element);
  void _doTestEmpty();
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
  m_queue = makeQueue(m_runner);
  _executeTest1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorLocalMemoryUnitTest::
_executeTest1()
{
  _doTestEmpty();

  // Tests with a number of blocks and a block size.
  _doTest(32, 149);
  _doTest(32 * 4, 137);
  _doTest(32 * 9, 275);
  _doTest(512, 311);
  _doTest(1024, 957);

  // Tests with a number of elements that is not a multiple of the block size.
  _doTest(0, 1023);
  _doTest(0, 1023 * 1023);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorLocalMemoryUnitTest::
_doTestEmpty()
{
  auto command = makeCommand(m_queue);
  ax::WorkGroupLoopRange loop_range;
  command << RUNCOMMAND_LAUNCH(work_group_context, loop_range)
  {
    ARCANE_UNUSED(work_group_context);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// If group_size==0, then nb_block_or_total_nb_element is the
// total number of elements. Otherwise, it is the number of blocks.

void AcceleratorLocalMemoryUnitTest::
_doTest(Int32 group_size, Int32 nb_group_or_total_nb_element)
{
  // Simple test of hierarchical parallelism and local memory usage.

  // All WorkItems in a group increment a counter
  // in shared memory. The last WorkItem in the group then copies
  // this array to global memory.

  info() << "DO_TEST group_size=" << group_size
         << " nb_group_or_total_nb_element=" << nb_group_or_total_nb_element;

  auto command = makeCommand(m_queue);

  ax::WorkGroupLoopRange<Int32> loop_range;
  if (group_size > 0) {
    Int32 total = group_size * nb_group_or_total_nb_element;
    loop_range = ax::WorkGroupLoopRange<Int32>(total);
    loop_range.setBlockSize(group_size);
  }
  else {
    loop_range = ax::WorkGroupLoopRange<Int32>(nb_group_or_total_nb_element);
    loop_range.setBlockSize(command);
  }

  const Int32 nb_group = loop_range.nbBlock();
  // NOTE: on accelerator, the size of a WorkGroup must be
  // a multiple of 32 and less than the maximum number of threads in a block
  // (generally 1024).
  info() << "DO_LOOP2 LocalMemory nb_group=" << nb_group
         << " group_size=" << group_size
         << " total_nb_element=" << loop_range.nbElement();

  ax::LocalMemory<Int64, 33> local_data_int64(command);
  ax::LocalMemory<Int32> local_data_int32(command, 50);
  const Int32 out_array_size = nb_group;

  NumArray<Int64, MDDim1> out_array(out_array_size);
  out_array.fillHost(0);
  auto out_span = viewInOut(command, out_array);

  // In multi-thread, selects the grain size to ensure multiple threads are used.
  if (m_queue.executionPolicy() == ax::eExecutionPolicy::Thread) {
    ParallelLoopOptions loop_options;
    loop_options.setGrainSize(nb_group / 4);
    command.setParallelLoopOptions(loop_options);
  }
  command << RUNCOMMAND_LAUNCH(context, loop_range, local_data_int32, local_data_int64)
  {
    auto work_block = context.block();
    auto work_item = context.workItem();
    auto local_span_int32 = local_data_int32.span();
    auto local_span_int64 = local_data_int64.span();

    // WorkItem 0 of the group initializes the shared memory
    const bool is_rank0 = (work_item.rankInBlock() == 0);
    if (is_rank0) {
      local_span_int32.fill(0);
      local_span_int64.fill(0);
    }

    // Ensures that all WorkItems in the block wait for initialization
    work_block.barrier();

    // Processes each loop index managed by the WorkItem.
    // It will add values to the shared memory.
    for (Int32 i : work_item.linearIndexes()) {
      ax::doAtomicAdd(&local_span_int32[i % local_span_int32.size()], 1);
      ax::doAtomicAdd(&local_span_int64[i % local_span_int64.size()], 10);
    }

    // To test 'constexpr' only on the device
    if constexpr (work_block.isDevice()) {
      if (is_rank0)
        ax::doAtomicAdd(&local_span_int32[0], 2);
    }

    // Ensures that all WorkItems have finished the atomic addition.
    work_block.barrier();

    // WorkItem 0 copies the shared array into the output array
    // at the index corresponding to its group.
    if (is_rank0) {
      Int32 group_index = work_block.groupRank();
      for (Int32 s : local_span_int32)
        out_span[group_index] += s;
      for (Int64 s : local_span_int64)
        out_span[group_index] += s;
    }
  };

  bool is_accelerator = m_queue.isAcceleratorPolicy();
  for (Int32 i = 0, n = out_array_size; i < n; ++i) {
    Int32 nb_active_item = loop_range.blockSize();
    // For the last block, the number of active elements is not necessarily group_size
    if ((i + 1) == n) {
      nb_active_item = (loop_range.nbElement() - (loop_range.blockSize() * (loop_range.nbBlock() - 1)));
    }
    Int64 out_value = out_span[i];
    const Int32 base_value = nb_active_item + nb_active_item * 10;
    // On accelerator, we add 2 because there is an addition in the lambda's 'constexpr' block.
    Int64 expected_value = (is_accelerator) ? (base_value + 2) : base_value;
    if (i < 10)
      info() << "DO_LOOP2 LocalMemory out[" << i << "]=" << out_value;
    if (out_value != expected_value)
      ARCANE_FATAL("Bad value for index '{0}' expected={1} v={2}", i, expected_value, out_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
