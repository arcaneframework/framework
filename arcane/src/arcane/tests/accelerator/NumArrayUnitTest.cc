// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayUnitTest.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Service de test des 'NumArray'.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/PointerAttribute.h"
#include "arcane/accelerator/core/ProfileRegion.h"
#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/SpanViews.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
extern "C++" ARCANE_CORE_EXPORT void
_arcaneTestRealArrayVariant();
extern "C++" ARCANE_CORE_EXPORT void
_arcaneTestRealArray2Variant();
} // namespace Arcane

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

namespace
{
inline MDSpan<double,MDDim1> _toMDSpan(const Span<double>& v)
{
  std::array<Int32,1> sizes = { (Int32)v.size() };
  return MDSpan<double,MDDim1>(v.data(),sizes);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'NumArray'.
 */
class NumArrayUnitTest
: public BasicUnitTest
{
 public:

  explicit NumArrayUnitTest(const ServiceBuildInfo& cb);
  ~NumArrayUnitTest();

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
  static constexpr double _getValue(Int64 i, Int64 j, Int64 k)
  {
    return static_cast<double>(i * 2 + j * 3 + k * 4);
  }
  static constexpr double _getValue(Int64 i, Int64 j, Int64 k, Int64 l)
  {
    return static_cast<double>(i * 2 + j * 3 + k * 4 + l * 8);
  }

  template <typename NumArrayType> double
  _doSum(const NumArrayType& values, std::array<Int32,NumArrayType::rank()> bounds, RunQueue* queue = nullptr)
  {
    if (queue)
      queue->barrier();
    constexpr int Rank = NumArrayType::rank();
    double total = 0.0;
    SimpleForLoopRanges<Rank> lb(bounds);
    arcaneSequentialFor(lb, [&](ArrayIndex<Rank> idx) { total += values(idx); });
    return total;
  }

 public:

  template<typename NumArrayType> void
  _doRank1(RunQueue& queue,NumArrayType& t1,Real expected_sum);
  template<typename NumArrayType> void
  _doRank2(RunQueue& queue,NumArrayType& t1,Real expected_sum);
  template<typename NumArrayType> void
  _doRank3(RunQueue& queue,NumArrayType& t1,Real expected_sum);
  template<typename NumArrayType> void
  _doRank4(RunQueue& queue,NumArrayType& t1,Real expected_sum);

 public:

  void _executeTest1(eMemoryRessource mem_kind);
  void _executeTest2();
  void _executeTest3();
  void _executeTest4(eMemoryRessource mem_kind);
  void _executeTest5();

 private:

  void _checkPointerAttribute(eMemoryRessource mem,const void* ptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(NumArrayUnitTest, IUnitTest, NumArrayUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NumArrayUnitTest::
NumArrayUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NumArrayUnitTest::
~NumArrayUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
initializeTest()
{
  IApplication* app = subDomain()->application();
  const auto& acc_info = app->acceleratorRuntimeInitialisationInfo();
  initializeRunner(m_runner, traceMng(), acc_info);
  Runner runner2(m_runner.executionPolicy());
  runner2.setAsCurrentDevice();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
executeTest()
{
  if (isAcceleratorPolicy(m_runner.executionPolicy())) {
    info() << "ExecuteTest1: using accelerator";
    _executeTest1(eMemoryRessource::UnifiedMemory);
    _executeTest1(eMemoryRessource::HostPinned);
    _executeTest1(eMemoryRessource::Device);

    info() << "ExecuteTest4: using accelerator";
    _executeTest4(eMemoryRessource::UnifiedMemory);
    _executeTest4(eMemoryRessource::HostPinned);
    _executeTest4(eMemoryRessource::Device);
  }
  else {
    info() << "ExecuteTest1: using host";
    _executeTest1(eMemoryRessource::Host);
    info() << "ExecuteTest4: using host";
    _executeTest4(eMemoryRessource::Host);
  }

  // Appelle deux fois _executeTest2() pour vérifier l'utilisation des pools
  // de RunQueue.
  _executeTest2();
  _executeTest2();
  _executeTest3();
  _executeTest5();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename NumArrayType> void NumArrayUnitTest::
_doRank2(RunQueue& queue,NumArrayType& t1,Real expected_sum)
{
  ValueChecker vc(A_FUNCINFO);

  auto command = makeCommand(queue);
  auto out_t1 = viewOut(command, t1);
  Int32 n1 = t1.extent0();
  Int32 n2 = t1.extent1();
  command << RUNCOMMAND_LOOP2(iter, n1, n2)
  {
    auto [i, j] = iter();
    out_t1(i, j) = _getValue(i, j);
  };
  NumArrayType host_t1(eMemoryRessource::Host);
  host_t1.copy(t1);
  double s2 = _doSum(host_t1, { n1, n2 });
  info() << "SUM2 = " << s2;
  vc.areEqual(s2, expected_sum, "SUM2");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename NumArrayType> void NumArrayUnitTest::
_doRank3(RunQueue& queue,NumArrayType& t1,Real expected_sum)
{
  ValueChecker vc(A_FUNCINFO);

  auto command = makeCommand(queue);
  auto out_t1 = viewOut(command, t1);
  Int32 n1 = t1.extent0();
  Int32 n2 = t1.extent1();
  Int32 n3 = t1.extent2();

  command << RUNCOMMAND_LOOP3(iter, n1, n2, n3)
  {
    auto [i, j, k] = iter();
    out_t1(i, j, k) = _getValue(i, j, k);
  };
  NumArrayType host_t1(eMemoryRessource::Host);
  host_t1.copy(t1);
  double s3 = _doSum(host_t1, { n1, n2, n3 });
  info() << "SUM3 = " << s3;
  vc.areEqual(s3, expected_sum, "SUM3");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename NumArrayType> void NumArrayUnitTest::
_doRank4(RunQueue& queue,NumArrayType& t1,Real expected_sum)
{
  ValueChecker vc(A_FUNCINFO);

  auto command = makeCommand(queue);
  auto out_t1 = viewOut(command, t1);
  Int32 n1 = t1.extent0();
  Int32 n2 = t1.extent1();
  Int32 n3 = t1.extent2();
  Int32 n4 = t1.extent3();
  info() << "SIZE = " << n1 << " " << n2 << " " << n3 << " " << n4;

  command << RUNCOMMAND_LOOP4(iter, n1, n2, n3, n4)
  {
    auto [i, j, k, l] = iter();
    out_t1(i, j, k, l) = _getValue(i, j, k, l);
  };
  NumArrayType host_t1(eMemoryRessource::Host);
  host_t1.copy(t1);
  double s4 = _doSum(host_t1, { n1, n2, n3, n4 });
  info() << "SUM4 = " << s4;
  vc.areEqual(s4, expected_sum, "SUM4");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
_executeTest1(eMemoryRessource mem_kind)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Test1 memory_ressource=" << mem_kind;

  auto queue = makeQueue(m_runner);
  Accelerator::ProfileRegion ps(queue,"NumArrayUniTest_Test1");

  // Ne pas changer les dimensions du tableau sinon
  // il faut aussi changer le calcul des sommes
  constexpr int n1 = 1000;
  constexpr int n2 = 3;
  constexpr int n3 = 4;
  constexpr int n4 = 13;

  constexpr double expected_sum1 = 999000.0;
  constexpr double expected_sum2 = 3006000.0;
  constexpr double expected_sum3 = 12096000.0;
  constexpr double expected_sum4 = 164736000.0;

  {
    NumArray<double, MDDim1> t1(mem_kind);
    t1.resize(n1);
    _checkPointerAttribute(mem_kind, t1.bytes().data());

    NumArray<double, MDDim1> t2(mem_kind);
    t2.resize(n1);

    NumArray<double, MDDim1> t3(mem_kind);
    t3.resize(n1);

    {
      [[maybe_unused]] auto span_value = t1.mdspan();
      using ValueType1 = NumArray<double, MDDim1>::value_type;
      using ValueType2 = decltype(span_value)::value_type;
      bool is_same_type = std::is_same_v<ValueType1, ValueType2>;
      std::cout << "IS_SAME: " << is_same_type << "\n";
      if (!is_same_type)
        ARCANE_FATAL("Not same value type");

      using LayoutType1 = NumArray<double, MDDim1>::LayoutPolicyType;
      using LayoutType2 = decltype(span_value)::LayoutPolicyType;
      bool is_same_policy = std::is_same_v<LayoutType1, LayoutType2>;
      if (!is_same_policy)
        ARCANE_FATAL("Not same policy");
    }
    {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, t1);
      command.addNbThreadPerBlock(128);
      if (command.nbThreadPerBlock() != 128)
        ARCANE_FATAL("Bad number of thread per block (v={0} expected=128)", command.nbThreadPerBlock());

      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        if ((i % 2) == 0)
          out_t1(i) = _getValue(i);
        else
          out_t1[i] = _getValue(i);
      };
      NumArray<double, MDDim1> host_t1(eMemoryRessource::Host);
      host_t1.copy(t1);
      double s1 = _doSum(host_t1, { n1 });
      info() << "SUM1 = " << s1;
      vc.areEqual(s1, expected_sum1, "SUM1");
    }
    {
      auto command = makeCommand(queue);
      auto in_t1 = t1.constMDSpan();
      MDSpan<double, MDDim1> out_t2 = t2.mdspan();

      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        auto span1 = in_t1.constSpan().to1DSpan();
        auto span2 = out_t2.to1DSpan();
        span2[i] = span1[i];
      };

      NumArray<double, MDDim1> host_t2(eMemoryRessource::Host);
      host_t2.copy(t2);
      double s2 = _doSum(host_t2, { n1 });
      info() << "SUM1_2 = " << s2;
      vc.areEqual(s2, expected_sum1, "SUM1_2");
    }
    {
      auto command = makeCommand(queue);
      ax::NumArrayInView<double, MDDim1> in_t1 = viewIn(command, t1);
      ax::NumArrayOutView<double, MDDim1> out_t3 = viewOut(command, t3);

      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        out_t3.to1DSpan()[i] = in_t1.to1DSpan()[i];
      };

      NumArray<double, MDDim1> host_t3(eMemoryRessource::Host);
      host_t3.copy(t3);
      double s3 = _doSum(host_t3, { n1 });
      info() << "SUM1_3 = " << s3;
      vc.areEqual(s3, expected_sum1, "SUM1_3");
    }
  }

  // Tableaux 2D
  {
    NumArray<double, MDDim2> t1(mem_kind);
    t1.resizeDestructive(n1, n2);
    _doRank2(queue, t1, expected_sum2);
  }
  {
    NumArray<double, ExtentsV<Int32, n1, n2>> t1(mem_kind);
    _doRank2(queue, t1, expected_sum2);
  }
  {
    NumArray<double, ExtentsV<Int32, DynExtent, n2>> t1(mem_kind);
    t1.resize(n1);
    _doRank2(queue, t1, expected_sum2);
  }
  {
    NumArray<double, ExtentsV<Int32, n1, DynExtent>> t1(mem_kind);
    t1.resize(n2);
    _doRank2(queue, t1, expected_sum2);
  }

  // Tableaux 3D
  {
    NumArray<double, MDDim3, LeftLayout> t1(mem_kind);
    t1.resizeDestructive(n1, n2, n3);
    _doRank3(queue, t1, expected_sum3);
  }
  {
    NumArray<double, MDDim3, RightLayout> t1(mem_kind);
    t1.resizeDestructive(n1, n2, n3);
    _doRank3(queue, t1, expected_sum3);
  }
  {
    NumArray<double, ExtentsV<Int32, DynExtent, n2, n3>, LeftLayout> t1(mem_kind);
    t1.resize(n1);
    _doRank3(queue, t1, expected_sum3);
  }
  {
    NumArray<double, ExtentsV<Int32, n1, n2, n3>, LeftLayout> t1(mem_kind);
    _doRank3(queue, t1, expected_sum3);
  }
  {
    NumArray<double, ExtentsV<Int32, DynExtent, n2, DynExtent>, LeftLayout> t1(mem_kind);
    t1.resize(n1, n3);
    _doRank3(queue, t1, expected_sum3);
  }

  // Tableaux 4D
  {
    NumArray<double, MDDim4> t1(mem_kind);
    t1.resize(n1, n2, n3, n4);
    _doRank4(queue, t1, expected_sum4);
  }
  {
    NumArray<double, ExtentsV<Int32, n1, DynExtent, DynExtent, n4>> t1(mem_kind);
    t1.resize(n2, n3);
    _doRank4(queue, t1, expected_sum4);
  }
  {
    NumArray<double, ExtentsV<Int32, DynExtent, DynExtent, n3, n4>, LeftLayout> t1(mem_kind);
    t1.resize(n1, n2);
    _doRank4(queue, t1, expected_sum4);
  }
  {
    NumArray<double, ExtentsV<Int32, n1, n2, n3, n4>> t1(mem_kind);
    _doRank4(queue, t1, expected_sum4);
  }

  {
    // Test copie.
    NumArray<double, MDDim1> t1(mem_kind);
    //t1.resize(22);
    t1.resize(n1);
    {
      auto command = makeCommand(queue);
      ax::NumArrayInOutView<double, MDDim1> out_t1 = viewInOut(command, t1);

      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        out_t1[iter] = _getValue(i);
      };
    }
    info() << "CHECK ALLOCATOR";
    NumArray<double, MDDim1> t2(t1);
    if (t1.memoryRessource() != t2.memoryRessource())
      ARCANE_FATAL("Bad memory ressource 1");
    if (t1.memoryAllocator() != t2.memoryAllocator())
      ARCANE_FATAL("Bad allocator 1");

    NumArray<double, MDDim1> host_t1(eMemoryRessource::Host);
    host_t1.copy(t2);
    double s3 = _doSum(host_t1, { n1 });
    info() << "SUM1_4 = " << s3;
    vc.areEqual(s3, expected_sum1, "SUM1_4");

    NumArray<double, MDDim1> t3;
    t3.resize(25);
    t3 = t1;
    if (t1.memoryRessource() != t3.memoryRessource())
      ARCANE_FATAL("Bad memory ressource 2 t1={0} t3={1}", t1.memoryRessource(), t3.memoryRessource());
    if (t1.memoryAllocator() != t3.memoryAllocator())
      ARCANE_FATAL("Bad allocator 2");
    NumArray<double, MDDim1> host_t3(eMemoryRessource::Host);
    host_t3.copy(t3);
    double s5 = _doSum(host_t3, { n1 });
    info() << "SUM1_5 = " << s5;
    vc.areEqual(s5, expected_sum1, "SUM1_5");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
_executeTest2()
{
  // Teste plusieurs queues simultanément.
  ValueChecker vc(A_FUNCINFO);

  // Ne pas changer les dimensions du tableau sinon
  // il faut aussi changer le calcul des sommes
  constexpr int n1 = 1000;
  constexpr int n2 = 3;
  constexpr int n3 = 4;
  constexpr int n4 = 13;

  constexpr double expected_sum4 = 164736000.0;

  // Test copie de Runner
  Runner runner1(m_runner);
  Runner runner2;
  runner2 = runner1;
  auto queue1 = makeQueue(m_runner);
  queue1.setAsync(true);
  auto queue2 = makeQueue(runner1);
  queue2.setAsync(true);
  auto queue3 = makeQueue(runner2);
  queue3.setAsync(true);

  NumArray<double, MDDim4> t1(n1, n2, n3, n4);

  // NOTE: Normalement il ne devrait pas être autorisé d'accéder au
  // même tableau depuis plusieurs commandes sur des files différentes
  // mais cela fonctionne avec la mémoire unifiée.

  // Utilise 3 files asynchrones pour positionner les valeurs du tableau,
  // chaque file gérant une partie du tableau.
  {
    auto command = makeCommand(queue1);
    auto out_t1 = viewOut(command, t1);
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue1,out_t1.to1DSpan().data());

    Int32 s1 = 300;
    auto b = makeLoopRanges(s1, n2, n3, n4);
    command << RUNCOMMAND_LOOP(iter, b)
    {
      auto [i, j, k, l] = iter();
      out_t1(i, j, k, l) = _getValue(i, j, k, l);
    };
  }
  {
    auto command = makeCommand(queue2);
    auto out_t1 = viewOut(command, t1);
    Int32 base = 300;
    Int32 s1 = 400;
    auto b = makeLoopRanges({ base, s1 }, n2, n3, n4);
    command << RUNCOMMAND_LOOP(iter, b)
    {
      auto [i, j, k, l] = iter();
      out_t1(i, j, k, l) = _getValue(i, j, k, l);
    };
  }
  {
    auto command = makeCommand(queue3);
    auto out_t1 = viewOut(command, t1);
    Int32 base = 700;
    Int32 s1 = 300;
    auto b = makeLoopRanges({ base, s1 }, n2, n3, n4);
    command << RUNCOMMAND_LOOP(iter, b)
    {
      auto [i, j, k, l] = iter();
      out_t1(i, j, k, l) = _getValue(i, j, k, l);
    };
  }
  queue1.barrier();
  queue2.barrier();
  queue3.barrier();

  double s4 = _doSum(t1, { n1, n2, n3, n4 });
  info() << "SUM4_ASYNC = " << s4;
  vc.areEqual(s4, expected_sum4, "SUM4_ASYNC");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
_executeTest3()
{
  Arcane::_arcaneTestRealArrayVariant();
  Arcane::_arcaneTestRealArray2Variant();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
_executeTest4(eMemoryRessource mem_kind)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Test4 memory_ressource=" << mem_kind;

  auto queue = makeQueue(m_runner);
  queue.setAsync(true);

  // Ne pas changer les dimensions du tableau sinon
  // il faut aussi changer le calcul des sommes
  constexpr int n1 = 1000;

  constexpr double expected_sum1 = 999000.0;
  IMemoryAllocator* allocator = MemoryUtils::getAllocator(mem_kind);

  {
    SharedArray<double> t1;
    {
      SharedArray<double> t1_bis(allocator);
      t1_bis.resize(n1);
      t1 = t1_bis;
    }

    SharedArray<double> t2(allocator);
    t2.resize(n1);

    SharedArray<double> t3(allocator);
    t3.resize(n1);

    {
      auto command = makeCommand(queue);
      auto out_t1 = viewOut(command, t1);

      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        if ((i % 2) == 0)
          out_t1(i) = _getValue(i);
        else
          out_t1[i] = _getValue(i);
      };
      NumArray<double, MDDim1> host_t1(eMemoryRessource::Host);
      host_t1.copy(_toMDSpan(t1), queue);
      double s1 = _doSum(host_t1, { n1 }, &queue);
      info() << "SUM1 = " << s1;
      vc.areEqual(s1, expected_sum1, "SUM1");
    }
    {
      auto command = makeCommand(queue);
      auto in_t1 = t1.constSpan();
      auto out_t2 = t2.span();

      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        auto span1 = in_t1;
        auto span2 = out_t2;
        span2[i] = span1[i];
      };

      NumArray<double, MDDim1> host_t2(eMemoryRessource::Host);
      host_t2.copy(_toMDSpan(t2), queue);
      double s2 = _doSum(host_t2, { n1 }, &queue);
      info() << "SUM1_2 = " << s2;
      vc.areEqual(s2, expected_sum1, "SUM1_2");
    }
    {
      auto command = makeCommand(queue);
      auto in_t1 = viewIn(command, t1);
      auto out_t3 = viewInOut(command, t3);

      command << RUNCOMMAND_LOOP1(iter, n1)
      {
        auto [i] = iter();
        out_t3[i] = in_t1[i];
      };

      NumArray<double, MDDim1> host_t3(eMemoryRessource::Host);
      host_t3.copy(_toMDSpan(t3), &queue);
      double s3 = _doSum(host_t3, { n1 }, &queue);
      info() << "SUM1_3 = " << s3;
      vc.areEqual(s3, expected_sum1, "SUM1_3");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
_executeTest5()
{
  ValueChecker vc(A_FUNCINFO);

  NumArray<Int32, MDDim1> t1;
  t1.setDebugName("Bonjour");
  String str_test = "Bonjour";
  vc.areEqual(t1.debugName(), str_test, "Pas bonjour :-(");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
void _checkMemoryType(ax::ePointerMemoryType expected,ax::ePointerMemoryType current)
{
  if (expected!=current)
    ARCANE_FATAL("Bad memory type expected={0} current={1}",(int)expected,(int)current);
}
void _checkNonNullHostPointer(const ax::PointerAttribute& pa)
{
  const void* p = pa.hostPointer();
  if (!p)
    ARCANE_FATAL("Host pointer is null");
}
void _checkNonNullDevicePointer(const ax::PointerAttribute& pa)
{
  const void* p = pa.devicePointer();
  if (!p)
    ARCANE_FATAL("Device pointer is null");
}
}
void NumArrayUnitTest::
_checkPointerAttribute(eMemoryRessource mem,const void* ptr)
{
  ValueChecker vc(A_FUNCINFO);
  ax::PointerAttribute pa;
  m_runner.fillPointerAttribute(pa,ptr);
  auto mem_type = pa.memoryType();
  info() << "PointerInfo mem_ressource=" << mem
         << " allocated_type=" << (int)mem_type
         << " host_ptr=" << pa.hostPointer()
         << " device_ptr=" << pa.devicePointer()
         << " device=" << pa.device()
         << " access_info=" << (int)getPointerAccessibility(&m_runner,ptr);
  if (mem==eMemoryRessource::UnifiedMemory){
    _checkMemoryType(mem_type,ax::ePointerMemoryType::Managed);
    _checkNonNullHostPointer(pa);
    _checkNonNullDevicePointer(pa);
  }
  if (mem==eMemoryRessource::HostPinned){
    _checkMemoryType(mem_type,ax::ePointerMemoryType::Host);
    _checkNonNullHostPointer(pa);
  }
  if (mem==eMemoryRessource::Device){
    _checkMemoryType(mem_type,ax::ePointerMemoryType::Device);
    _checkNonNullDevicePointer(pa);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Exemples d'utilisation de NumArray pour la documentation
extern "C" void
_arcaneNumArraySamples()
{
  //![SampleNumArrayDeclarations]
  Arcane::NumArray<double, Arcane::MDDim1> a1; //< Tableau dynamique de rang 1.
  Arcane::NumArray<double, Arcane::MDDim2> a2; //< Tableau dynamique de rang 2.
  Arcane::NumArray<double, Arcane::MDDim3> a3; //< Tableau dynamique de rang 3.
  Arcane::NumArray<double, Arcane::MDDim4> a4; //< Tableau dynamique de rang 4.
  //![SampleNumArrayDeclarations]

  //![SampleNumArrayDeclarationsExtented]
  // Tableau 2D avec 2 dimensions statiques (3x4)
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, 3, 4>> t1;
  // Tableau 2D avec la première dimension dynamique et la deuxième fixée à 5.
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, Arcane::DynExtent, 5>> t2;
  // Tableau 3D avec la première dimension et troisième dimension dynamique et la deuxième fixée à 5.
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, Arcane::DynExtent, 7, Arcane::DynExtent>> t3;
  //![SampleNumArrayDeclarationsExtented]

  //![SampleNumArrayResize]
  // Tableau 2D avec 2 dimensions statiques (3x4)
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, 3, 4>> t4;
  // Tableau 2D avec 8x5 valeurs
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, Arcane::DynExtent, 5>> t5(8);
  // Tableau 3D avec 3x7x9 valeurs
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, Arcane::DynExtent, 7, Arcane::DynExtent>> t6(3, 9);
  // Redimensionne t2 avec 6x5 valeurs
  t2.resize(6);
  // Redimensionne t3 avec 2x7x4 valeurs
  t3.resizeDestructive(2, 4);
  // Redimensionne a1 avec 12 valeurs
  a1.resizeDestructive(12);
  // Redimensionne a2 avec 3x4 valeurs
  a2.resizeDestructive(3, 4);
  // Redimensionne a3 avec 2x7x6 valeurs
  a3.resizeDestructive(2, 7, 6);
  // Redimensionne a1 avec 2x9x4x6 valeurs
  a4.resizeDestructive(2, 9, 4, 6);
  //![SampleNumArrayResize]

  //![SampleNumArrayDeclarationsMemory]
  // Tableau 2D avec 2 dimensions statiques (3x4) sur l'accélérateur
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, 3, 4>> t7(Arcane::eMemoryRessource::Device);
  // Tableau 2D avec 8x5 valeurs alloué sur l'hôte.
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, Arcane::DynExtent, 5>> t8(8, Arcane::eMemoryRessource::Host);
  // Tableau 3D avec 3x7x9 valeurs allouée sur le device.
  Arcane::NumArray<double, Arcane::ExtentsV<Arcane::Int32, Arcane::DynExtent, 7, Arcane::DynExtent>> t9(3, 9, Arcane::eMemoryRessource::Device);
  //![SampleNumArrayDeclarationsMemory]

  //![SampleNumArrayDeclarationsIndexation]
  Arcane::ArrayIndex<1> i1(2);
  Arcane::ArrayIndex<2> i2(2, 3);
  Arcane::ArrayIndex<3> i3(1, 6, 5);
  Arcane::ArrayIndex<4> i4(1, 7, 2, 4);
  a1(i1) = 3.0;
  a1(2) = 4.0;
  a2(i2) = 2.0;
  a2(2, 3) = 4.0;
  a3(i3) = 1.0;
  a3(1, 6, 5) = 6.0;
  a4(i4) = 1.0;
  a4(1, 7, 2, 4) = 6.0;
  //![SampleNumArrayDeclarationsIndexation]

  {
    Arcane::NumArray<Arcane::Int32, Arcane::MDDim2> aaa(5, 4);
    Arcane::MDSpan<Arcane::Int32, Arcane::MDDim1> bbb(aaa.mdspan().slice(0));

    Arcane::MDSpan<Arcane::Real, Arcane::MDDim3> a3_span(a3.mdspan());
    Arcane::MDSpan<Arcane::Real, Arcane::MDDim1> a2_span1(a2.mdspan().slice(0));
    for (Arcane::Int32 i = 0; i < a3.extent0(); ++i) {
      Arcane::MDSpan<Arcane::Real, Arcane::MDDim2> span_array2 = a3_span.slice(i);
      std::cout << " MDDim2 slice i=" << i << " X=" << span_array2.extent0()
                << " Y=" << span_array2.extent1() << "\n";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
