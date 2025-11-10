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
#include "arcane/accelerator/RunCommandLocalMemory.h"
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorLocalMemoryUnitTest::
_executeTest1()
{
  auto queue = makeQueue(m_runner);

  // Test simple du parallélisme hiérarchique et de l'utilisation
  // de la mémoire locale.

  // Tous les WorkItem d'un groupe incrémentent un compteur
  // en mémoire partagé. Le dernier WorkItem du groupe recopie
  // ensuite ce tableau en mémoire locale.

  // NOTE: pour l'instant ce test suppose que la taille
  // d'un WorkGroup est fixée en dur à 256 qui doit aussi être le
  // nombre de thread par bloc sur accélérateur.
  {
    Int32 loop_size = 1024 * 1024;
    info() << "DO_LOOP2 LocalMemory size=" << loop_size;
    auto command = makeCommand(queue);
    ax::RunCommandLocalMemory<Int32> local_data(command, 50);
    const Int32 out_array_size = loop_size / 256;
    NumArray<Int32, MDDim1> out_array(out_array_size);
    out_array.fillHost(0);
    auto out_span = viewInOut(command, out_array);
    Accelerator::Impl::WorkGroupLoopRange loop_range(loop_size);
    command << RUNCOMMAND_LOOP(work_item, loop_range, local_data)
    {
      Int32 group_index = work_item.groupRank();
      auto s = local_data.span();
      if (work_item.rankInGroup() == 0)
        for (int j = 0; j < 50; ++j)
          s[j] = 0;

      work_item.barrier();
      Int32 i = work_item();
      ax::doAtomicAdd(&s[i % 50], 1);

      work_item.barrier();

      // Le dernier élément du groupe recopie dans le tableau de sortie.
      // On prend le dernier pour que cela fonctionne correctement en séquentiel
      if (work_item.rankInGroup() == (work_item.groupSize()-1)) {
        for (int j = 0; j < 50; ++j)
          out_span[group_index] += s[j];
      }
    };

    for (Int32 i = 0, n = out_array_size; i < n; ++i) {
      Int32 out_value = out_span[i];
      if (i<10)
        info() << "DO_LOOP2 LocalMemory out[" << i << "]=" << out_value;      
      if (out_value != 256)
        ARCANE_FATAL("Bad value for index '{0}' expected=256 v={1}", i, out_value);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
