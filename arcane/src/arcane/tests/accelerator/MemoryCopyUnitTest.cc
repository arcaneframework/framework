// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopyUnitTest.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Service de test des noyaux de recopie mémoire.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"
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
class MemoryCopyUnitTest
: public BasicUnitTest
{
 public:

  explicit MemoryCopyUnitTest(const ServiceBuildInfo& cb);
  ~MemoryCopyUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

  static constexpr double _getValue(Int64 i)
  {
    return static_cast<double>(i * 2);
  }
  static constexpr double _getValue(Int64 i, Int64 j)
  {
    return static_cast<double>(i * 2 + j * 3);
  }

 public:

  void _executeTest1(eMemoryRessource mem_kind, bool use_queue = true);
  void _executeCopy(eMemoryRessource mem_kind, bool use_queue);
  void _executeFill(eMemoryRessource mem_kind, bool use_queue, bool use_index);
  void _fillIndexes(Int32 n1, NumArray<Int32, MDDim1>& indexes);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(MemoryCopyUnitTest, IUnitTest, MemoryCopyUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryCopyUnitTest::
MemoryCopyUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryCopyUnitTest::
~MemoryCopyUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryCopyUnitTest::
initializeTest()
{
  m_runner = *(subDomain()->acceleratorMng()->defaultRunner());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryCopyUnitTest::
executeTest()
{
  if (ax::impl::isAcceleratorPolicy(m_runner.executionPolicy())) {
    info() << "ExecuteTest1: using accelerator";
    _executeTest1(eMemoryRessource::UnifiedMemory);
    _executeTest1(eMemoryRessource::HostPinned);
    _executeTest1(eMemoryRessource::Device);
  }
  else {
    info() << "ExecuteTest1: using host";
    _executeTest1(eMemoryRessource::Host, false);
    _executeTest1(eMemoryRessource::Host, true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryCopyUnitTest::
_executeTest1(eMemoryRessource mem_kind, bool use_queue)
{
  _executeCopy(mem_kind, use_queue);
  _executeFill(mem_kind, use_queue, false);
  _executeFill(mem_kind, use_queue, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryCopyUnitTest::
_fillIndexes(Int32 n1, NumArray<Int32, MDDim1>& indexes)
{
  Int32 nb_index = 0;
  {
    UniqueArray<Int32> indexes_buf;
    int mod = 3;
    for (int i = 0; i < n1; ++i) {
      if ((i % mod) == 0) {
        indexes_buf.add(i);
        ++mod;
        if (mod > 8)
          mod /= 2;
      }
    }
    nb_index = indexes_buf.size();
    info() << "NB_INDEX=" << nb_index;
    indexes.resize(nb_index);
    for (int i = 0; i < nb_index; ++i)
      indexes[i] = indexes_buf[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryCopyUnitTest::
_executeCopy(eMemoryRessource mem_kind, bool use_queue)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Copy memory_ressource=" << mem_kind << " use_queue=" << use_queue;

  auto queue = makeQueue(m_runner);
  RunQueue* queue_ptr = &queue;
  if (!use_queue)
    queue_ptr = nullptr;

  constexpr int n1 = 2500;
  constexpr int n2 = 4;

  NumArray<Int32, MDDim1> indexes;
  _fillIndexes(n1, indexes);
  Int32 nb_index = indexes.dim1Size();

  info() << "Test Rank1";
  {
    NumArray<double, MDDim1> t1(mem_kind);
    t1.resize(n1);
    NumArray<double, MDDim1> destination_buffer(mem_kind);
    destination_buffer.resize(nb_index);

    {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, t1);

      // Remplit t1 avec les bonnes valeurs
      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        out_t1(iter) = _getValue(i);
      };
    }

    {
      ConstMemoryView source(t1.to1DSpan());
      MutableMemoryView destination(destination_buffer.to1DSpan());
      MemoryUtils::copyFromIndexes(destination, source, indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      MemoryUtils::copyFromIndexes(destination, source, {}, queue_ptr);
    }

    NumArray<double, MDDim1> host_destination(eMemoryRessource::Host);
    host_destination.copy(destination_buffer);

    {
      NumArray<double, MDDim1> host_t1(eMemoryRessource::Host);
      host_t1.copy(t1);

      for (int i = 0; i < nb_index; ++i) {
        auto v1 = host_destination(i);
        auto v2 = host_t1(indexes(i));
        if (v1 != v2) {
          ARCANE_FATAL("Bad copy from i={0} v1={1} v2={2}", i, v1, v2);
        }
      }
    }

    // Remet des valeurs fausses dans t1
    {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, t1);
      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        out_t1(iter) = -1.0;
      };
    }

    {
      MutableMemoryView t1_view(t1.to1DSpan());
      ConstMemoryView destination_view(destination_buffer.to1DSpan());
      MemoryUtils::copyToIndexes(t1_view, destination_view,indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      MemoryUtils::copyToIndexes(t1_view, destination_view, {}, queue_ptr);
    }

    // Vérifie la validité
    {
      NumArray<double, MDDim1> host_t1(eMemoryRessource::Host);
      host_t1.copy(t1);

      for (int i = 0; i < nb_index; ++i) {
        auto v1 = host_destination(i);
        auto v2 = host_t1(indexes(i));
        if (v1 != v2) {
          ARCANE_FATAL("Bad copy to i={0} v1={1} v2={2}", i, v1, v2);
        }
      }
    }
  }
  info() << "End of test Rank1";

  info() << "Test Rank2 CopyFrom";
  {
    NumArray<double, MDDim2> t1(mem_kind);
    t1.resize(n1, n2);
    NumArray<double, MDDim2> destination_buffer(mem_kind);
    destination_buffer.resize(nb_index, n2);

    {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, t1);

      command << RUNCOMMAND_LOOP2(iter, n1, n2)
      {
        auto [i, j] = iter();
        out_t1(i, j) = _getValue(i, j);
      };
    }

    {
      ConstMemoryView source(t1.to1DSpan(), n2);
      MutableMemoryView destination(destination_buffer.to1DSpan(), n2);
      MemoryUtils::copyFromIndexes(destination, source, indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      MemoryUtils::copyFromIndexes(destination, source, {}, queue_ptr);
    }

    NumArray<double, MDDim2> host_destination(eMemoryRessource::Host);
    host_destination.copy(destination_buffer);

    {
      NumArray<double, MDDim2> host_t1(eMemoryRessource::Host);
      host_t1.copy(t1);

      for (int i = 0; i < nb_index; ++i) {
        for (int j = 0; j < n2; ++j) {
          auto v1 = host_destination(i, j);
          auto v2 = host_t1(indexes(i), j);
          if (v1 != v2) {
            ARCANE_FATAL("Bad copy from i={0} j={1} v1={2} v2={3}", i, j, v1, v2);
          }
        }
      }
    }

    // Remet des valeurs fausses dans t1
    {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, t1);
      command << RUNCOMMAND_LOOP2(iter, n1, n2)
      {
        out_t1(iter) = -1.0;
      };
    }

    {
      MutableMemoryView t1_view(t1.to1DSpan(), n2);
      ConstMemoryView source_view(destination_buffer.to1DSpan(), n2);
      MemoryUtils::copyToIndexes(t1_view, source_view, indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      MemoryUtils::copyToIndexes(t1_view, source_view, {}, queue_ptr);
    }

    {
      NumArray<double, MDDim2> host_t1(eMemoryRessource::Host);
      host_t1.copy(t1);

      for (int i = 0; i < nb_index; ++i) {
        for (int j = 0; j < n2; ++j) {
          auto v1 = host_destination(i, j);
          auto v2 = host_t1(indexes(i), j);
          if (v1 != v2) {
            ARCANE_FATAL("Bad copy to i={0} j={1} v1={2} v2={3}", i, j, v1, v2);
          }
        }
      }
    }
  }
  info() << "End of test Rank2 OK!";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryCopyUnitTest::
_executeFill(eMemoryRessource mem_kind, bool use_queue, bool use_index)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Fill memory_ressource=" << mem_kind
         << " use_queue=" << use_queue << " use_index=" << use_index;

  RunQueue queue = makeQueue(m_runner);
  RunQueue* queue_ptr = &queue;
  if (!use_queue)
    queue_ptr = nullptr;

  constexpr int n1 = 2500;

  NumArray<Int32, MDDim1> indexes;
  if (use_index)
    _fillIndexes(n1, indexes);
  Int32 nb_index = indexes.dim1Size();

  info() << "Test Rank1";
  {
    NumArray<double, MDDim1> t1(mem_kind);
    t1.resize(n1);
    NumArray<double, MDDim1> destination_buffer(mem_kind);
    destination_buffer.resize(nb_index);

    {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, t1);

      // Remplit t1 avec les bonnes valeurs
      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        out_t1(iter) = _getValue(i);
      };
    }

    // TODO: Ajoute test avec d'autres types (par exemple Real3)
    // TODO: Faire des tests asynchrones

    const double fill_value = 3.4;
    if (use_index)
      t1.fill(fill_value, indexes, queue_ptr);
    else
      t1.fill(fill_value, queue_ptr);

    NumArray<double, MDDim1> host_t1(eMemoryRessource::Host);
    host_t1.copy(t1);

    {
      // Regarde si les valeurs correspondantes aux index sont correctes
      Int32 nb_to_test = (use_index) ? nb_index : n1;
      for (Int32 i = 0; i < nb_to_test; ++i) {
        auto v1 = fill_value;
        Int32 index = (use_index) ? indexes(i) : i;
        auto v2 = host_t1(index);
        if (v1 != v2) {
          ARCANE_FATAL("Bad fill from i={0} index={1} v1={2} v2={3}", i, index, v1, v2);
        }
      }
    }
    if (use_index) {
      NumArray<Int16, MDDim1> filter(eMemoryRessource::Host);
      filter.resize(n1);
      filter.fillHost(0);
      for (Int32 i = 0; i < nb_index; ++i)
        filter[indexes[i]] = 1;

      // Regarde si les valeurs qui ne correspondent pas aux index sont correctes
      for (int i = 0; i < n1; ++i) {
        if (filter[i] == 0) {
          auto v1 = _getValue(i);
          auto v2 = host_t1(i);
          if (v1 != v2) {
            ARCANE_FATAL("Bad no fill from i={0} v1={1} v2={2}", i, v1, v2);
          }
        }
      }
    }
  }
  info() << "End of test Rank1";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
