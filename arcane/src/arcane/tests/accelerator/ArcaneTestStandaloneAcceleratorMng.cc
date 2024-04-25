// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/launcher/ArcaneLauncher.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunQueue.h"
#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

void _printAllocator(IMemoryRessourceMng* mrm, eMemoryRessource r)
{
  IMemoryAllocator* allocator = mrm->getAllocator(r, false);
  std::cout << "Allocator name=" << r << " v=" << (allocator != nullptr) << "\n";
}

void _printAvailableAllocators()
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  _printAllocator(mrm, eMemoryRessource::Host);
  _printAllocator(mrm, eMemoryRessource::Device);
}

void _testSum(IAcceleratorMng* acc_mng)
{
  // Test la somme de deux tableaux 'a' et 'b' dans un tableau 'c'.

  int nb_value = 10000;
  NumArray<Int64, MDDim1> a(nb_value);
  NumArray<Int64, MDDim1> b(nb_value);
  NumArray<Int64, MDDim1> c(nb_value);
  for (int i = 0; i < nb_value; ++i) {
    a(i) = i + 2;
    b(i) = i + 3;
  }

  {
    auto command = makeCommand(acc_mng->defaultQueue());
    auto in_a = viewIn(command, a);
    auto in_b = viewIn(command, b);
    auto out_c = viewOut(command, c);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      out_c(i) = in_a(i) + in_b(i);
    };
  }

  Int64 total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += c(i);
  std::cout << "TOTAL=" << total << "\n";
  Int64 expected_total = 100040000;
  if (total != expected_total)
    ARCANE_FATAL("Bad value for sum={0} (expected={1})", total, expected_total);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _testBinOp(IAcceleratorMng* acc_mng)
{
  // Test des opérateurs binaires pour les views

  int nb_value = 10000;
  NumArray<Int64, MDDim1> a(nb_value);
  NumArray<Int64, MDDim1> b(nb_value);
  for (int i = 0; i < nb_value; ++i) {
    a(i) = 1;
    b(i) = 2;
  }

  // *=
  {
    auto command = makeCommand(acc_mng->defaultQueue());
    auto in_out_a = viewInOut(command, a);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      in_out_a(i) *= 2.;
    };
  }
  Int64 total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += a(i);
  std::cout << "TOTAL=" << total << "\n";
  Int64 expected_total = nb_value * 2;
  if (total != expected_total)
    ARCANE_FATAL("Bad value for operator*= {0} (expected={1})", total, expected_total);

  // +=
  {
    auto command = makeCommand(acc_mng->defaultQueue());
    auto out_a = viewOut(command, a);
    auto in_b = viewIn(command, b);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      out_a(i) += in_b(i);
    };
  }
  total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += a(i);
  std::cout << "TOTAL=" << total << "\n";
  expected_total = nb_value * 4;
  if (total != expected_total)
    ARCANE_FATAL("Bad value for operator+= {0} (expected={1})", total, expected_total);

  // -= et /=
  {
    auto command = makeCommand(acc_mng->defaultQueue());
    auto in_out_a = viewInOut(command, a);
    auto in_b = viewIn(command, b);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      in_out_a(i) -= in_b(i);
      in_out_a(i) /= 2;
    };
  }
  total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += a(i);
  std::cout << "TOTAL=" << total << "\n";
  expected_total = nb_value;
  if (total != expected_total)
    ARCANE_FATAL("Bad value {0} (expected={1})", total, expected_total);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _testEmptyKernel(IAcceleratorMng* acc_mng)
{
  // Lance un kernel vide pour évaluer le coup du lancement.
  int nb_value = 2000;
  int nb_iteration = 10000;

  auto queue = Accelerator::makeQueue(acc_mng->defaultRunner());
  queue.setAsync(true);
  Int64 xbegin = platform::getRealTimeNS();
  for (int i = 0; i < nb_iteration; ++i) {
    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(, nb_value){};
  }
  Int64 xend = platform::getRealTimeNS();
  queue.barrier();
  Int64 xend2 = platform::getRealTimeNS();
  std::cout << "Time1 = " << (xend - xbegin) / nb_iteration << " Time2=" << (xend2 - xbegin) / nb_iteration << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int _testStandaloneLauncher(const CommandLineArguments& cmd_line_args,
                            const String& method_name)
{
  ARCANE_UNUSED(method_name);
  ArcaneLauncher::init(cmd_line_args);
  StandaloneAcceleratorMng launcher{ ArcaneLauncher::createStandaloneAcceleratorMng() };
  IAcceleratorMng* acc_mng = launcher.acceleratorMng();
  RunQueue* default_queue = acc_mng->defaultQueue();
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  IMemoryAllocator* alloc0 = MemoryUtils::getDeviceOrHostAllocator();
  IMemoryAllocator* wanted_alloc = mrm->getAllocator(eMemoryRessource::Host);
  if (default_queue->isAcceleratorPolicy())
    wanted_alloc = mrm->getAllocator(eMemoryRessource::Device);
  if (alloc0 != wanted_alloc)
    ARCANE_FATAL("Bad allocator");

  _printAvailableAllocators();
  if (method_name == "TestSum")
    _testSum(acc_mng);
  else if (method_name == "TestBinOp")
    _testBinOp(acc_mng);
  else if (method_name == "TestEmptyKernel")
    _testEmptyKernel(acc_mng);
  else
    ARCANE_FATAL("Unknown method to test");
  return 0;
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_EXPORT int
arcaneTestStandaloneAcceleratorLauncher(const CommandLineArguments& cmd_line_args,
                                        const String& method_name)
{
  int r =0;
  try{
    r = _testStandaloneLauncher(cmd_line_args,method_name);
  }
  catch(const Exception& ex){
    std::cerr << "EXCEPTION: " << ex << "\n";
    throw;
  }
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
