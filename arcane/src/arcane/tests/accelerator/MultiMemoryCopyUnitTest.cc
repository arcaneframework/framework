// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiMemoryCopyUnitTest.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Service de test des noyaux de recopie mémoire.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryView.h"

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
 * \brief Service de test des noyaux de recopie mémoire.
 */
class MultiMemoryCopyUnitTest
: public BasicUnitTest
{
 public:

  explicit MultiMemoryCopyUnitTest(const ServiceBuildInfo& cb);
  ~MultiMemoryCopyUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;
  NumArray<Int32, MDDim1> m_indexes;

 private:

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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(MultiMemoryCopyUnitTest, IUnitTest, MultiMemoryCopyUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMemoryCopyUnitTest::
MultiMemoryCopyUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiMemoryCopyUnitTest::
~MultiMemoryCopyUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMemoryCopyUnitTest::
initializeTest()
{
  m_runner = *(subDomain()->acceleratorMng()->defaultRunner());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiMemoryCopyUnitTest::
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

void MultiMemoryCopyUnitTest::
_executeTest1(eMemoryRessource mem_kind, bool use_queue)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Test2 memory_ressource=" << mem_kind << " use_queue=" << use_queue;

  auto queue = makeQueue(m_runner);
  RunQueue* queue_ptr = &queue;
  if (!use_queue)
    queue_ptr = nullptr;

  constexpr int n2 = 4;

  constexpr int nb_dim2 = 7;
  constexpr std::array<Int32, nb_dim2> nb_element_by_dim = { 2300, 134, 459, 12903, 15, 550, 9670 };

  // Pour tester la copie avec MultiMemory, le tableau d'indices
  // contient pour les index pairs le numéro de la mémoire et
  // les pour les index impair l'indice dans le tableau correspondant
  Int32 nb_index = 0;
  {
    UniqueArray<Int32> indexes_buf;
    int mod = 3;
    for (int zz = 0; zz < nb_dim2; ++zz) {
      const Int32 n1 = nb_element_by_dim[zz];
      for (int i = 0; i < n1; ++i) {
        if ((i % mod) == 0) {
          Int32 array_index = zz;
          Int32 value_index = i % nb_element_by_dim[array_index];
          indexes_buf.add(array_index);
          indexes_buf.add(value_index);
          ++mod;
          if (mod > 8)
            mod /= 2;
        }
      }
    }
    nb_index = indexes_buf.size() / 2;
    info() << "NB_INDEX=" << nb_index;
    m_indexes.resize(nb_index * 2);
    for (int i = 0; i < (nb_index * 2); ++i)
      m_indexes[i] = indexes_buf[i];
  }

  info() << "Test Rank1";
  {
    // Buffer contenant les valeurs sérialisées
    NumArray<double, MDDim1> buffer(mem_kind);
    // Permet de vérifier que tout les autres NumArray ont bien le bon allocateur
    IMemoryAllocator* allocator = buffer.memoryAllocator();
    buffer.resize(nb_index);

    UniqueArray<NumArray<double, MDDim1>> memories;
    for (int i = 0; i < nb_dim2; ++i)
      memories.add(NumArray<double, MDDim1>(mem_kind));
    for (int i = 0; i < nb_dim2; ++i) {
      memories[i].resize(nb_element_by_dim[i]);
      IMemoryAllocator* a2 = memories[i].memoryAllocator();
      if (a2 != allocator)
        ARCANE_FATAL("Bad allocator");
    }

    // Attention à ne plus modifier les dimensions dans 'memories'
    // après avoir récupérer les 'Span'
    UniqueArray<Span<std::byte>> memories_as_bytes(platform::getDefaultDataAllocator(), nb_dim2);
    for (int i = 0; i < nb_dim2; ++i)
      memories_as_bytes[i] = memories[i].bytes();

    for (int zz = 0; zz < nb_dim2; ++zz) {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, memories[zz]);
      const Int32 n = memories[zz].extent0();
      command << RUNCOMMAND_LOOP1(iter, n)
      {
        auto [i] = iter();
        out_t1(iter) = _getValue(i);
      };
    }

    // Effectue la copie dans le buffer
    {
      MultiConstMemoryView source_memory_view(memories_as_bytes.constView(), sizeof(double));
      MutableMemoryView destination(buffer.to1DSpan());
      source_memory_view.copyToIndexes(destination, m_indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      source_memory_view.copyToIndexes(destination, {}, queue_ptr);
    }

    UniqueArray<NumArray<double, MDDim1>> host_memories;
    for (int i = 0; i < nb_dim2; ++i)
      host_memories.add(NumArray<double, MDDim1>(eMemoryRessource::Host));
    for (int i = 0; i < nb_dim2; ++i) {
      host_memories[i].resize(nb_element_by_dim[i]);
      host_memories[i].copy(memories[i]);
    }

    NumArray<double, MDDim1> host_buffer(eMemoryRessource::Host);
    host_buffer.copy(buffer);

    // Vérifie le résultat
    info() << "Check result copyTo";
    for (int i = 0; i < nb_index; ++i) {
      Int32 array_index = m_indexes[(i * 2)];
      Int32 value_index = m_indexes[(i * 2) + 1];
      auto v1 = host_buffer(i);
      auto v2 = host_memories[array_index][value_index];
      if (v1 != v2) {
        ARCANE_FATAL("Bad copy from i={0} v1={1} v2={2}", i, v1, v2);
      }
    }

    // Remet des valeurs fausses dans t1
    for (int zz = 0; zz < nb_dim2; ++zz) {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, memories[zz]);
      const Int32 n = memories[zz].extent0();
      command << RUNCOMMAND_LOOP1(iter, n)
      {
        out_t1(iter) = -1.0;
      };
    }

    // Effectue la copie depuis le buffer
    {
      MultiMutableMemoryView source_memory_view(memories_as_bytes.view(), sizeof(double));
      ConstMemoryView source(buffer.to1DSpan());
      source_memory_view.copyFromIndexes(source, m_indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      source_memory_view.copyFromIndexes(source, {}, queue_ptr);
    }

    // Vérifie le résultat
    info() << "Check result copyFrom";
    for (int i = 0; i < nb_index; ++i) {
      Int32 array_index = m_indexes[(i * 2)];
      Int32 value_index = m_indexes[(i * 2) + 1];
      auto v1 = host_buffer(i);
      auto v2 = host_memories[array_index][value_index];
      if (v1 != v2) {
        ARCANE_FATAL("Bad copy to i={0} v1={1} v2={2}", i, v1, v2);
      }
    }
  }
  info() << "End of test Rank1";

  info() << "Test Rank2";
  {
    // Buffer contenant les valeurs sérialisées
    NumArray<double, MDDim2> buffer(mem_kind);
    // Permet de vérifier que tout les autres NumArray ont bien le bon allocateur
    IMemoryAllocator* allocator = buffer.memoryAllocator();
    buffer.resize(nb_index, n2);

    UniqueArray<NumArray<double, MDDim2>> memories;
    for (int i = 0; i < nb_dim2; ++i)
      memories.add(NumArray<double, MDDim2>(mem_kind));
    for (int i = 0; i < nb_dim2; ++i) {
      memories[i].resize(nb_element_by_dim[i], n2);
      IMemoryAllocator* a2 = memories[i].memoryAllocator();
      if (a2 != allocator)
        ARCANE_FATAL("Bad allocator");
    }

    // Attention à ne plus modifier les dimensions dans 'memories'
    // après avoir récupérer les 'Span'
    UniqueArray<Span<std::byte>> memories_as_bytes(platform::getDefaultDataAllocator(), nb_dim2);
    for (int i = 0; i < nb_dim2; ++i)
      memories_as_bytes[i] = memories[i].bytes();

    for (int zz = 0; zz < nb_dim2; ++zz) {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, memories[zz]);
      const Int32 n = memories[zz].extent0();
      command << RUNCOMMAND_LOOP2(iter, n, n2)
      {
        auto [i, j] = iter();
        out_t1(iter) = _getValue(i, j);
      };
    }

    // Effectue la copie dans le buffer
    {
      MultiConstMemoryView source_memory_view(memories_as_bytes.constView(), sizeof(double) * n2);
      MutableMemoryView destination(buffer.to1DSpan(), n2);
      source_memory_view.copyToIndexes(destination, m_indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      source_memory_view.copyToIndexes(destination, {}, queue_ptr);
    }

    UniqueArray<NumArray<double, MDDim2>> host_memories;
    for (int i = 0; i < nb_dim2; ++i)
      host_memories.add(NumArray<double, MDDim2>(eMemoryRessource::Host));
    for (int i = 0; i < nb_dim2; ++i) {
      host_memories[i].resize(nb_element_by_dim[i], n2);
      host_memories[i].copy(memories[i]);
    }

    NumArray<double, MDDim2> host_buffer(eMemoryRessource::Host);
    host_buffer.copy(buffer);

    // Vérifie le résultat
    info() << "Check result copyTo";
    for (int i = 0; i < nb_index; ++i) {
      Int32 array_index = m_indexes[(i * 2)];
      Int32 value_index = m_indexes[(i * 2) + 1];
      for (int j = 0; j < n2; ++j) {
        auto v1 = host_buffer(i, j);
        auto v2 = host_memories[array_index](value_index, j);
        if (v1 != v2) {
          ARCANE_FATAL("Bad copy from i={0} j={1} v1={2} v2={3}", i, j, v1, v2);
        }
      }
    }

    // Remet des valeurs fausses dans t1
    for (int zz = 0; zz < nb_dim2; ++zz) {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, memories[zz]);
      const Int32 n = memories[zz].extent0();
      command << RUNCOMMAND_LOOP2(iter, n, n2)
      {
        out_t1(iter) = -1.0;
      };
    }

    // Effectue la copie depuis le buffer
    {
      MultiMutableMemoryView source_memory_view(memories_as_bytes.view(), sizeof(double) * n2);
      ConstMemoryView source(buffer.to1DSpan(), n2);
      source_memory_view.copyFromIndexes(source, m_indexes.to1DSpan().smallView(), queue_ptr);
      // Teste copie vide
      source_memory_view.copyFromIndexes(source, {}, queue_ptr);
    }

    // Vérifie le résultat
    info() << "Check result copyFrom";
    for (int i = 0; i < nb_index; ++i) {
      Int32 array_index = m_indexes[(i * 2)];
      Int32 value_index = m_indexes[(i * 2) + 1];
      for (int j = 0; j < n2; ++j) {
        auto v1 = host_buffer(i, j);
        auto v2 = host_memories[array_index](value_index, j);
        if (v1 != v2) {
          ARCANE_FATAL("Bad copy to i={0} v1={1} v2={2}", i, v1, v2);
        }
      }
    }
  }
  info() << "End of test Rank2";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
