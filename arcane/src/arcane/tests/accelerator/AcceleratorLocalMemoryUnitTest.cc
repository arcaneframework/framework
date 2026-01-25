// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorLocalMemoryUnitTest.cc                           (C) 2000-2026 */
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

  // Tests avec un nombre de bloc et une taille d'un bloc.
  _doTest(32, 149);
  _doTest(32 * 4, 137);
  _doTest(32 * 9, 275);
  _doTest(512, 311);
  _doTest(1024, 957);

  // Tests avec un nombre d'éléments qui n'est pas un multiple de la taille d'un bloc.
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

// Si \a group_size==0, alors nb_block_or_total_nb_element est le
// nombre total d'éléments. Sinon il s'agit du nombre de bloc.

void AcceleratorLocalMemoryUnitTest::
_doTest(Int32 group_size, Int32 nb_group_or_total_nb_element)
{
  // Test simple du parallélisme hiérarchique et de l'utilisation
  // de la mémoire locale.

  // Tous les WorkItem d'un groupe incrémentent un compteur
  // en mémoire partagé. Le dernier WorkItem du groupe recopie
  // ensuite ce tableau en mémoire globale.

  info() << "DO_TEST group_size=" << group_size
         << " nb_group_or_total_nb_element=" << nb_group_or_total_nb_element;

  auto command = makeCommand(m_queue);

  ax::WorkGroupLoopRange<Int32> loop_range;
  if (group_size > 0){
    Int32 total = group_size * nb_group_or_total_nb_element;
    loop_range = ax::WorkGroupLoopRange<Int32>(total);
    loop_range.setBlockSize(group_size);
  }
  else {
    loop_range = ax::WorkGroupLoopRange<Int32>(nb_group_or_total_nb_element);
    loop_range.setBlockSize(command);
  }

  const Int32 nb_group = loop_range.nbBlock();
  // NOTE: sur accélérateur, la taille d'un WorkGroup doit être
  // un multiple de 32 et inférieur au nombre maximum de thread d'un bloc
  // (en général 1024).
  info() << "DO_LOOP2 LocalMemory nb_group=" << nb_group
         << " group_size=" << group_size
         << " total_nb_element=" << loop_range.nbElement();

  ax::LocalMemory<Int64, 33> local_data_int64(command);
  ax::LocalMemory<Int32> local_data_int32(command, 50);
  const Int32 out_array_size = nb_group;

  NumArray<Int64, MDDim1> out_array(out_array_size);
  out_array.fillHost(0);
  auto out_span = viewInOut(command, out_array);

  // En multi-thread, sélectionne la taille de grain pour être sur
  // d'utiliser plusieurs threads.
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

    // Le WorkItem 0 du groupe initialise la mémoire partagée
    const bool is_rank0 = (work_item.rankInBlock() == 0);
    if (is_rank0) {
      local_span_int32.fill(0);
      local_span_int64.fill(0);
    }

    // S'assure que tous les WorkItem du bloc attendent l'initialisation
    work_block.barrier();

    // Traite chaque indice de la boucle géré par le WorkItem.
    // Il va ajouter des valeurs à la mémoire partagée.
    for ( Int32 i : work_item.linearIndexes() ) {
      ax::doAtomicAdd(&local_span_int32[i % local_span_int32.size()], 1);
      ax::doAtomicAdd(&local_span_int64[i % local_span_int64.size()], 10);
    }

    // Pour tester le 'constexpr' uniquement sur le device
    if constexpr (work_block.isDevice()) {
      if (is_rank0)
        ax::doAtomicAdd(&local_span_int32[0], 2);
    }

    // S'assure que tous les WorkItem ont terminé l'ajout atomique.
    work_block.barrier();

    // Le WorkItem 0 recopie le tableau partagé dans le tableau de sortie
    // à l'indice correspondant à son groupe.
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
    // Pour le dernier bloc, le nombre d'éléments actif n'est pas forcément group_size
    if ((i + 1) == n) {
      nb_active_item = (loop_range.nbElement() - (loop_range.blockSize() * (loop_range.nbBlock() - 1)));
    }
    Int64 out_value = out_span[i];
    const Int32 base_value = nb_active_item + nb_active_item * 10;
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
