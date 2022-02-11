// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayUnitTest.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Service de test des 'NumArray'.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"

#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/Runner.h"
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

  template <int Rank> double
  _doSum(NumArray<double, Rank> values, ArrayBounds<Rank> bounds)
  {
    double total = 0.0;
    SimpleLoopRanges<Rank> lb(bounds);
    arcaneSequentialFor(lb, [&](ArrayBoundsIndex<Rank> idx) { total += values(idx); });
    return total;
  }

 public:

  void _executeTest1(eMemoryRessource mem_kind);
  void _executeTest2();
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NumArrayUnitTest::
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
    _executeTest1(eMemoryRessource::Host);
  }

  // Appelle deux fois _executeTest2() pour vérifier l'utilisation des pools
  // de RunQueue.
  _executeTest2();
  _executeTest2();
}

void NumArrayUnitTest::
_executeTest1(eMemoryRessource mem_kind)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Test1 memory_ressource=" << mem_kind;

  auto queue = makeQueue(m_runner);
  auto command = makeCommand(queue);

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
    NumArray<double, 1> t1(mem_kind);
    t1.resize(n1);

    auto out_t1 = viewOut(command, t1);

    command << RUNCOMMAND_LOOP1(iter, n1)
    {
      auto [i] = iter();
      out_t1(i) = _getValue(i);
    };
    NumArray<double, 1> host_t1(eMemoryRessource::Host);
    host_t1.copy(t1);
    double s1 = _doSum(host_t1, { n1 });
    info() << "SUM1 = " << s1;
    vc.areEqual(s1, expected_sum1, "SUM1");
  }

  {
    NumArray<double, 2> t1(mem_kind);
    t1.resize(n1, n2);

    auto out_t1 = viewOut(command, t1);

    command << RUNCOMMAND_LOOP2(iter, n1, n2)
    {
      auto [i, j] = iter();
      out_t1(i, j) = _getValue(i, j);
    };
    NumArray<double, 2> host_t1(eMemoryRessource::Host);
    host_t1.copy(t1);
    double s2 = _doSum(host_t1, { n1, n2 });
    info() << "SUM2 = " << s2;
    vc.areEqual(s2, expected_sum2, "SUM2");
  }

  {
    NumArray<double, 3> t1(mem_kind);
    t1.resize(n1, n2, n3);

    auto out_t1 = viewOut(command, t1);

    command << RUNCOMMAND_LOOP3(iter, n1, n2, n3)
    {
      auto [i, j, k] = iter();
      out_t1(i, j, k) = _getValue(i, j, k);
    };
    NumArray<double, 3> host_t1(eMemoryRessource::Host);
    host_t1.copy(t1);
    double s3 = _doSum(host_t1, { n1, n2, n3 });
    info() << "SUM3 = " << s3;
    vc.areEqual(s3, expected_sum3, "SUM3");
  }

  {
    NumArray<double, 4> t1(mem_kind);
    t1.resize(n1, n2, n3, n4);

    auto out_t1 = viewOut(command, t1);

    command << RUNCOMMAND_LOOP4(iter, n1, n2, n3, n4)
    {
      auto [i, j, k, l] = iter();
      out_t1(i, j, k, l) = _getValue(i, j, k, l);
    };
    NumArray<double, 4> host_t1(eMemoryRessource::Host);
    host_t1.copy(t1);
    double s4 = _doSum(host_t1, { n1, n2, n3, n4 });
    info() << "SUM4 = " << s4;
    vc.areEqual(s4, expected_sum4, "SUM4");
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

  auto queue1 = makeQueue(m_runner);
  queue1.setAsync(true);
  auto queue2 = makeQueue(m_runner);
  queue2.setAsync(true);
  auto queue3 = makeQueue(m_runner);
  queue3.setAsync(true);

  NumArray<double, 4> t1(n1, n2, n3, n4);

  // NOTE: Normalement il ne devrait pas être autorisé d'accéder au
  // même tableau depuis plusieurs commandes sur des files différentes
  // mais cela fonctionne avec la mémoire unifiée.

  // Utilise 3 files asynchrones pour positionner les valeurs du tableau,
  // chaque file gérant une partie du tableau.
  {
    auto command = makeCommand(queue1);
    auto out_t1 = viewOut(command, t1);
    Int64 s1 = 300;
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
    Int64 base = 300;
    Int64 s1 = 400;
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
    Int64 base = 700;
    Int64 s1 = 300;
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

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
