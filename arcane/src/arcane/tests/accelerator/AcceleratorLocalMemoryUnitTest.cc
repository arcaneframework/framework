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
#include "arcane/accelerator/WorkGroupLoopRange.h"

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
  RunQueue m_queue;

 public:

  void _executeTest1();
  void _doTest(Int32 block_size, Int32 nb_block);
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
  _doTest(32, 149);
  _doTest(32 * 4, 137);
  _doTest(32 * 9, 275);
  _doTest(512, 311);
  _doTest(1024, 957);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorLocalMemoryUnitTest::
_doTest(Int32 block_size, Int32 nb_block)
{
  // Test simple du parallélisme hiérarchique et de l'utilisation
  // de la mémoire locale.

  // Tous les WorkItem d'un groupe incrémentent un compteur
  // en mémoire partagé. Le dernier WorkItem du groupe recopie
  // ensuite ce tableau en mémoire globale.

  // NOTE: sur accélérateur, la taille d'un WorkGroup doit être
  // un multiple de 32 et inférieur au nombre maximum de thread d'un bloc
  // (en général 1024).
  info() << "DO_LOOP2 LocalMemory nb_block=" << nb_block << " block_size=" << block_size;

  auto command = makeCommand(m_queue);
  ax::LocalMemory<Int64, 33> local_data_int64(command, 33);
  ax::LocalMemory<Int32> local_data_int32(command, 50);
  const Int32 out_array_size = nb_block;

  NumArray<Int64, MDDim1> out_array(out_array_size);
  out_array.fillHost(0);
  auto out_span = viewInOut(command, out_array);

  ax::WorkGroupLoopRange loop_range(command, nb_block, block_size);

  command << RUNCOMMAND_LAUNCH(work_group, loop_range, local_data_int32, local_data_int64)
  {
    auto local_span_int32 = local_data_int32.span();
    auto local_span_int64 = local_data_int64.span();
    auto work_item0 = work_group.item0();
    if (work_item0.rankInGroup() == 0) {
      local_span_int32.fill(0);
      local_span_int64.fill(0);
    }

    work_item0.sync();

    // Sur accélérateur, nbItem() vaut toujours 1.
    for (Int32 g = 0; g < work_group.nbItem(); ++g) {
      auto work_item = work_group.item(g);
      Int32 i = work_item();
      ax::doAtomicAdd(&local_span_int32[i % local_span_int32.size()], 1);
      ax::doAtomicAdd(&local_span_int32[i % local_span_int64.size()], 10);
      if constexpr (work_item.isDevice()) {
        // Pour tester le 'constexpr' uniquement sur le device
        auto xy = work_item.x();
        int xy2 = xy;
        if (work_item.rankInGroup() == 0)
          ax::doAtomicAdd(&local_span_int32[0], xy2);
      }
    }

    work_item0.sync();

    // Recopie le tableau partagé dans le tableau de sortie.
    if (work_item0.rankInGroup() == 0) {
      Int32 group_index = work_item0.groupRank();
      for (Int32 s : local_span_int32)
        out_span[group_index] += s;
      for (Int64 s : local_span_int64)
        out_span[group_index] += s;
    }
  };

  bool is_accelerator = m_queue.isAcceleratorPolicy();
  for (Int32 i = 0, n = out_array_size; i < n; ++i) {
    Int64 out_value = out_span[i];
    const Int32 base_value = block_size + block_size * 10;
    // Sur accélérateur on ajoute 2 car il y a un ajout dans le 'constexpr' de la lambda
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
